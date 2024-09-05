import torch
import torch.nn as nn
from swin_tranformer.embedding import PatchMerge, Pathches
from swin_tranformer.encoder import Stage, MPLBlock


class SwinBase(nn.Module):
    """
    """
    def __init__(self,
                 in_chans=3, image_size=224, patch_size=4, ape=True,
                 num_classes=1000, depths=[2, 2, 6, 2], embed_dim=96,
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4,qkv_bias=True,
                 qk_scale=None, drop_out=0.1, norm_eps=1e-12, use_res_pos=True):
        super().__init__()
        self.in_chans = in_chans
        self.image_size = image_size
        self.num_heads = num_heads
        self.num_patches = (image_size // patch_size) ** 2
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_stages = len(depths)
        self.last_embed_dim = int(self.embed_dim * 2 ** (self.num_stage - 1)) # embed_dim at last stage

        # image size after embedding 
        self.patches_resolution = self.image_size // patch_size

        self.embedding = Pathches(kernel_size=patch_size, stride=patch_size, padding=None,
                                  in_chans=in_chans, embed_dim=self.embed_dim, norm_layer=True)

        # adding position embedding
        if self.ape is not None:
            self.absolute_pos_embed =  nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim), requires_grad = True)
        
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            stage = Stage(depth=self.depths[i],
                          embed_dim= int(embed_dim * 2 **i),
                          input_size= int(self.patches_resolution // 2 ** i),
                          num_heads= self.num_heads[i],
                          window_size= window_size, mlp_ratio= mlp_ratio, 
                          patches_merge=True if (i < self.num_stages - 1) else None, # merge patch at stage 1 2 3
                          qk_scale= qk_scale, qkv_bias=qkv_bias,
                          use_res_pos=use_res_pos, drop_out=drop_out, norm_eps=norm_eps)
            
            self.stages.append(stage) 
        
        self.norm = nn.LayerNorm(self.last_embed_dim)

        self.avgpool = nn.AvgPool1d(1)

        self.head = nn.Linear(in_features=self.last_embed_dim, out_features= self.num_classes)
        
    
    def forward(self, x: torch.Tensor):
        
        # Embedding
        x = self.embeding(x) # B, H*W, embed_dim

        if self.ape is not None:
            x = x + self.absolute_pos_embed  # B, H*W, embed_dim
        
        # Stage
        for stg in self.stages:
            x = stg(x)
        
        x = self.norm(x)
        x = self.avgpool(x.transpose(1,2)) # B, C, 1

        # head
        x = torch.flatten(x)
        x = self.head(x)
        return x