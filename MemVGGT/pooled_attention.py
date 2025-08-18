import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np  
import math

from utils import load_batches
from attention import MLP, Encoder, VisionTransformer
from patch_embed import ClassToken, PatchEmbed, PositionalEncoding

class PooledEncoder(nn.Module):
    """Transformer encoder block with spatiotemporal pooling."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_pool_stride: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        kv_cls_flag: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,  # expects (B, N, D)
        )
        self.kv_pool_stride = kv_pool_stride
        self.kv_cls_flag = kv_cls_flag
        self.drop_path1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6) # Pre-norm before MLP branch.
        self.mlp = MLP(embed_dim, mlp_ratio=4.0, dropout=dropout) # 2-layer FFN (expansion 4x).
        self.drop_path2 = nn.Dropout(dropout)   # Dropout on the MLP residual branch.

    @staticmethod
    def _pool_kv(x_spatial: torch.Tensor, hw: tuple[int, int, int], stride: int) -> torch.Tensor:
        """Pool key and value tensors."""

        if stride == 1:
            return x_spatial
            
        B, N, C = x_spatial.shape
        T, H, W = hw
        assert N == T * H * W, "Input feature has wrong size"

        # Reshape to (B, T, H, W, C) for pooling
        x3d = x_spatial.view(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        x3d_pooled = nn.AvgPool3d(kernel_size=(1, stride, stride), stride=(1, stride, stride))(x3d)
        _, C_pool, T_pool, H_pool, W_pool = x3d_pooled.shape
            
        # Reshape back to (B, N', C) where N' is the pooled sequence length
        x_pooled = x3d_pooled.permute(0, 2, 3, 4, 1).contiguous().view(B, T_pool * H_pool * W_pool, C_pool)
        return x_pooled
        
    def forward(self, x: torch.Tensor, hw: tuple[int, int, int]) -> torch.Tensor:
        
        """
        x: (B, S, C) where S = 1 + T*H*W if CLS present, else S = T*H*W
        thw: (T, H, W) for the current token grid
        returns: (B, S, C)  (same sequence length; KV pooling only)
        """

        B, S, C = x.shape
        T, H, W = hw
        has_cls = S == 1 + T * H * W

        x_norm = self.norm1(x)

        if has_cls:
            x_cls, x_spatial = x_norm[:, :1, :], x_norm[:, 1:, :]  # (B, 1, C), (B, T*H*W, C)
        else:
            x_cls, x_spatial = None, x_norm  # (B, T*H*W, C)
            
        # Pool key and value tensors if stride > 1
        if self.kv_pool_stride > 1:
            x_spatial_pooled = self._pool_kv(x_spatial, hw, self.kv_pool_stride)  # (B, T*H'*W', C)
        else:
            x_spatial_pooled = x_spatial

        if has_cls and self.kv_cls_flag:
            # Append CLS token to pooled key and value tensors
            x_kv = torch.cat((x_cls, x_spatial_pooled), dim=1)  # (B, 1 + T*H'*W', C)
        else:
            x_kv = x_spatial_pooled  # (B, T*H'*W', C)
            
        # Multi-head self-attention (pre-norm)
        x_attn, _ = self.attn(x_norm, x_kv, x_kv, need_weights=True)
        x = x + self.drop_path1(x_attn)  # Residual connection
        
        # MLP (pre-norm)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x
    
class PooledVisionTransformer(nn.Module):
    """
    Minimal Vision Transformer for image classification.

    Args:
        img: a sample image tensor to infer H,W for num_patches (kept to match your PatchEmbed API)
        batch_size: unused in model logic but kept to match your API
        patch_size: patch size (int)
        channels: input channels (e.g., 3)
        embed_dim: token dimension (D)
        num_heads: attention heads
        num_layers: number of encoder blocks
        num_classes: output classes; if None, returns CLS features
        dropout: dropout used in MLP and residual paths
        attn_dropout: attention dropout
    """
    def __init__(
        self,
        img: torch.Tensor,
        batch_size: int,
        patch_size: int,
        channels: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int | None = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        # Patchify -> tokens
        self.patch_embed = PatchEmbed(img, batch_size, patch_size, channels, embed_dim)
        self.class_token = ClassToken(embed_dim)

        # +1 for CLS token
        self.positional_encoding = PositionalEncoding(embed_dim, num_patches=1 + self.patch_embed.num_patches)

        # Precompute (T,H,W) for an image-only case (T=1)
        # H_img, W_img = img.shape[-2], img.shape[-1]
        # self.hw = (1, H_img // patch_size, W_img // patch_size)

        # Encoder stack
        self.blocks = nn.ModuleList(
            [
                PooledEncoder(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    kv_pool_stride = 1,
                    kv_cls_flag = True
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Classification head (optional)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes is not None else nn.Identity()

        # Save a few attributes (not strictly needed for forward)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Returns:
            If num_classes is set: (B, num_classes)
            Else: CLS features (B, D)
        """

        B, C, H_img, W_img = x.shape
        T = 1
        H = H_img // self.patch_size
        W = W_img // self.patch_size
        hw = (T, H, W)
        # 1) Patchify & project -> (B, L, D)

        x = self.patch_embed(x)

        # 2) Prepend CLS token -> (B, 1+L, D)
        x = self.class_token(x)

        # 3) Add absolute positional encodings
        #    (PositionalEncoding should slice to current length automatically)
        x = self.positional_encoding(x)

        # S = x.shape[1]
        # L = S - 1
        # T = 1  # Assuming single frame input; can be generalized later
        # H = W = int(math.sqrt(L))
        # hw = (T, H, W)  # Current token grid size

        # 4) Encoder blocks
        for blk in self.blocks:
            x = blk(x, hw)

        # 5) Final norm
        x = self.norm(x)

        # 6) Take CLS token (index 0)
        cls = x[:, 0, :]  # (B, D)

        # 7) Head
        out = self.head(cls)  # (B, num_classes) or identity (B, D)
        return out
    

def __main__():
    img_folder = '/home/sbangal4/world_in_3d/VGGT/trials/vggt/data/data_co3d/apple/110_13051_23361/images'
    images = load_batches(img_folder, limit=5)
    torch.manual_seed(0)
    model = PooledVisionTransformer(
        img=images,
        batch_size=5,
        patch_size=16,
        channels=3,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        num_classes=1000,
        dropout=0.1,
        attn_dropout=0.1,
    )
    logits = model(images)
    print(logits.shape)  # Should print (2, 1000)
    print(logits)
    print(logits.dtype)

if __name__ == "__main__":
    __main__()