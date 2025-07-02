
"""
Implementação do modelo HeartBEiT (Vision Transformer para ECG)
Baseado em BEiT (Baidu's Enhanced Image Transformer)
"""

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.layers import trunc_normal_

from .base_model import BaseModel
from ..config.model_configs import HeartBEiTConfig


class HeartBEiT(BaseModel):
    """HeartBEiT: Vision Transformer para sinais ECG"""
    
    def __init__(
        self,
        config: HeartBEiTConfig,
        img_size: int = 5000,  # Comprimento do sinal ECG
        in_chans: int = 12,    # Número de derivações
        num_classes: int = 5,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs
    ):
        super().__init__(num_classes=num_classes, input_channels=in_chans)
        self.config = config
        
        self.num_features = self.embed_dim = config.embed_dim
        
        # Patch embedding para sinais 1D
        # Adaptando PatchEmbed de timm para 1D
        # Assumimos que o sinal é (batch_size, in_chans, img_size)
        # Queremos transformar (in_chans, img_size) em uma sequência de patches
        
        # Uma abordagem é tratar cada derivação como um 'canal' e o comprimento como 'largura'
        # E então usar um PatchEmbed 2D com kernel_size=(1, patch_size)
        # Ou, mais diretamente, um conv1d para criar os patches
        
        # Vamos criar um PatchEmbed 1D customizado
        self.patch_embed = nn.Conv1d(
            in_chans, 
            config.embed_dim, 
            kernel_size=config.patch_size, 
            stride=config.patch_size
        )
        num_patches = img_size // config.patch_size
        self.num_patches = num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, config.depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(config.depth)
        ])
        self.norm = norm_layer(config.embed_dim)
        
        # Classifier head
        self.head = nn.Linear(config.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # x: (B, C, L) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x



