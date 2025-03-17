# tranformers block
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformer.embedding import PatchMerge

class Stage(nn.Module):
    """
    depth: number of SwinBlock in stage
    embed_dim: number of channels 

    """
    def __init__(self,
                 depth: int,
                 embed_dim: int,
                 input_size: int,
                 num_heads: int,
                 window_size: int,
                 mlp_ratio: float,
                 patches_merge=None,
                 qkv_bias=True,
                 qk_scale=None,
                 use_rel_pos=True,
                 drop_out=0.1,
                 norm_eps=1e-12):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.num_heads = num_heads
        self.patches_merge = patches_merge
        # build blocks
        self.blocks = nn.ModuleList([
            SwinBlock(embed_dim=self.embed_dim, num_heads=self.num_heads,
                      input_size=self.input_size, window_size=window_size,
                      shift_size=0 if (i % 2 == 0) else window_size // 2,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop_out=drop_out, norm_eps=norm_eps, use_rel_pos=use_rel_pos)
            for i in range(depth)
        ])

        if self.patches_merge is not None:
            self.merging = PatchMerge(embed_dim=self.embed_dim, image_size=(self.input_size, self.input_size),
                                 norm_layer=True)
    
    def forward(self, x: torch.Tensor):
        
        for bkl in self.blocks:
            x = bkl(x)

        if self.patches_merge is not None:
            x = self.merging(x)
            
        return x    


class MPLBlock(nn.Module):
    """
    """
    def __init__(self,
                 embed_dim,
                 mlp_dim,
                 drop_out= 0.1,
                 norm_eps = 1e-12):
        super().__init__()

        self.fc1 = nn.Linear(in_features=embed_dim, out_features=mlp_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=drop_out)
        self.fc2 =  nn.Linear(in_features=mlp_dim, out_features=embed_dim)
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x 


class SwinBlock(nn.Module):
    """
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 input_size: int,
                 window_size: int,
                 shift_size: int,
                 mlp_ratio: float,
                 qkv_bias: bool,
                 qk_scale: bool,
                 use_rel_pos: bool,
                 drop_out: float,
                 norm_eps = 1e-12,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.input_size = input_size
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn_mask = None

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MPLBlock(embed_dim= self.embed_dim, mlp_dim= mlp_hidden_dim, drop_out= drop_out, norm_eps= norm_eps)

        if self.input_size <= self.window_size:
            self.shift_size = 0
            self.window_size = self.input_size
        
        self.attention = WindowAttention(embed_dim= self.embed_dim,
                                         num_heads= self.num_heads,
                                         window_size= self.window_size,
                                         qk_scale= qk_scale,
                                         qkv_bias=qkv_bias,
                                         use_rel_pos=use_rel_pos)
        
        if self.shift_size > 0:
            H, W = self.input_size, self.input_size
            img_mask = torch.zeros(1, H, W, 1)

            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None)) 
            
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_window, _ = window_partition(img_mask, self.window_size)
            mask_window = mask_window.view(-1, self.window_size * self.window_size)
            attn_mask = mask_window.unsqueeze(1) - mask_window.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

            self.attn_mask = attn_mask

        self.norm = nn.LayerNorm(self.embed_dim, eps=norm_eps)


    def forward(self, x: torch.Tensor):
        H, W = self.input_size, self.input_size
        B, L, C = x.shape # B H*W, embed_dim

        assert L == H * W
        shortcut = x

        x = self.norm(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
            
            x_window, pad_hw = window_partition(shifted_x, self.window_size) # (B*n_windows, window_size, window_size, embed_dim)
        
        else:
            shifted_x = x
            x_window, pad_hw = window_partition(shifted_x, self.window_size) # (B*n_windows, window_size, window_size, embed_dim)
        
        x_window = x_window.view(-1, self.window_size * self.window_size, C)

        # attention
        attn_window = self.attention(x_window)

        attn_window = attn_window.view(-1, self.window_size, self.window_size, C) # (B*n_windows, window_size, window_size, embed_dim)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_unpartition(attn_window, self.window_size, pad_hw= pad_hw, hw = (H,W)) # (B, H, W, embed_dim)
        
        x = shifted_x.view(B, L, C)
        x += shortcut

        # FFN
        x = self.mlp(self.norm(x))

        return x


class WindowAttention(nn.Module):
    """
        embed_dim: number of input channels
        num_heads: number of attention heads
        input_size: height/ width of inputs for calculating the relative position parameter
        qkv_bias: if True, add learnable bias to q, k, v
        use_rel_pos: if True, use relative position embeddings to attention map

    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 window_size: int,
                 qk_scale=None,
                 qkv_bias=True,
                 use_rel_pos=True,
                 **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        
        self.num_heads = num_heads

        self.window_size = window_size

        head_dim = embed_dim // num_heads
        self.qk_scale = head_dim ** 0.5 if qk_scale == "None" else qk_scale

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * window_size - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * window_size - 1, head_dim))

        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias = qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor, mask = None):
        B, L, C = x.shape # B H*W embed_dim
        H,W = self.window_size, self.window_size

        # qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) # B 
        # q,k,v = qkv.reshape(3, B * self.num_heads, L, -1).unbind(0)

        qkv = self.qkv(x)
        q,k,v = tuple(rearrange(qkv, "b l (d f k) -> k (b f) l d", k=3, f=self.num_heads))

        #qk_dot_product = (q * self.qk_scale) @ k.transpose(-2, -1)
        qk_dot_product = torch.einsum("b i d, b j d -> b i j", q, k) * float(self.qk_scale)

        if self.use_rel_pos:
            attn = add_decompose_rel_pos(qk_dot_product, q, self.rel_pos_h, self.rel_pos_w, (H,W), (H,W)) # B* numhead, H*W, H*W
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, L, L) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B, L, L)

            attn = attn.softmax(dim=-1) 

        attn = attn.softmax(dim=-1)

        #x = torch.einsum("b i j, b j d -> b i d", attn, v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, L, C)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, L, C)
        x = self.proj(x)
        # output shape: B, H*W, embed_dim
        return x


def window_partition(x, window_size):
    """
        x: (B, H, W, C)
    """

    B,H,W,C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    output = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    
    return output, (Hp, Wp)


def window_unpartition(x, window_size, pad_hw: tuple, hw: tuple):
    """
        x: (B, window_size, window_size, C)
        pad_hw: (Hp, Wp)
        hw: size of input before partition
    """

    h, w = hw
    Hp, Wp = pad_hw

    B = x.shape[0] // (Hp * Wp // window_size // window_size)
    x = x.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > h or Wp > w:
        x = x[:, :h, :w, :].contiguous()
    
    return x


def get_rel_pos(q_size: int, k_size:int, rel_pos: torch.Tensor):
    """
        q_h: size of query 
        k_h: size of key 
        rel_pos_h: relative positional of height
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)

    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                                        size = max_rel_dist,
                                        mode = "linear")
        
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1,0)
    
    else:
        rel_pos_resized = rel_pos
    
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)

    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decompose_rel_pos(attn: torch.Tensor,
                          q: torch.Tensor,
                          rel_pos_h: torch.Tensor,
                          rel_pos_w: torch.Tensor,
                          q_size: tuple,
                          k_size: tuple,):
    """
        attn: attention map q @ k (shape: B* num_heads, H*W, H*W)
        q: query q in the attention layer (B * num_heads, H*W, embed_dim)
        rel_pos_h: relative position embedding for height axis
        rel_pos_w: relative position embedding for width axis
    """

    B, _, dim = q.shape
    q_h, q_w = q_size[0], q_size[1]
    k_h, k_w = k_size[0], k_size[1]

    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    r_q = q.view(B, q_h, q_w, dim).contiguous()
    rel_h = torch.einsum("bhwc, hkc -> bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc, wkc -> bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :,]
            ).view(B, q_h * q_w, k_h * k_w)
    
    return attn