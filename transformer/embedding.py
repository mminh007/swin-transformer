# patch embedding
import torch 
import torch.nn as nn

class Patches(nn.Module):
    """
    """
    def __init__(self,
                 kernel_size = 4,
                 stride = 4,
                 padding = 0,
                 in_chans = 3,
                 embed_dim = 96,
                 norm_layer = None):
        
        super().__init__()
        
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=kernel_size, stride=stride)
        
        if norm_layer is not None:
            self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):

        x = self.proj(x)
        B, C, H, W  = x.shape

        x = x.permute(0,2,3,1).contiguous().view(B, -1, self.embed_dim) # B, embed_dim,H,W -> B,H,W,embed_dim -> B, H*W, embed_dim
        
        if self.norm is not None:
            x = self.norm(x)

        return x


class PatchMerge(nn.Module):
    """
    embed_dim: number of channels
    norm_layer: Normalization
    image_size: size of input
    -------------------------
    Input:
        tensor (B, L, C)
    
    Output:
        tensor: (B, H/2 * W/2, 4*C)
    """

    def __init__(self,
                 embed_dim: int,
                 image_size: tuple,
                 norm_layer = None):
        super().__init__()
        
        self.img_size = image_size #(H,W)
        self.embed_dim = embed_dim
        
        # if norm_layer is not None:
        #     self.norm = nn.LayerNorm(self.embed_dim * 2)
        
        # self.downsample = nn.Conv2d(in_channels= embed_dim, out_channels= embed_dim * 2, kernel_size=2, stride=2, padding=0)

        self.reduce = nn.Linear(4 * embed_dim, 2 * embed_dim)
        if norm_layer is not None:
            self.norm = nn.LayerNorm(4 * embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, L, C = x.shape  # B, H*W, embed_dim
        
        H, W = self.img_size[0], self.img_size[1]

        #x = x.view(B, H, W, C).contiguous().permute(0,3,1,2)

        # downsample 
        # x = self.downsample(x)

        # x = x.permute(0,2,3,1).contiguous().view(B, -1, C * 2)  # B, h, w, embed_dim *2 -> B, h*w, embed_dim*2
 
        # if self.norm is not None:
        #     x = self.norm(x)

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2]
        x1 = x[:, 1::2, 0::2]
        x2 = x[:, 0::2, 1::2]
        x3 = x[:, 1::2, 1::2]

        x = torch.cat([x0, x1, x2, x3], dim = -1) 
        x = x.view(B, -1, 4 * C)

        if self.norm is not None:
            x = self.norm(x)
        
        x = self.reduce(x)
        
        return x
    
