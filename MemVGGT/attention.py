import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from utils import load_batches
from patch_embed import PatchEmbed, ClassToken, PositionalEncoding

class MLP(nn.Module):
    """2-layer feedforward head used inside each Transformer block."""
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Encoder(nn.Module):
    """Transformer encoder block."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,  # expects (B, N, D)
        )
        self.drop_path1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(embed_dim, mlp_ratio=4.0, dropout=dropout)
        self.drop_path2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention (pre-norm)
        x = x + self.drop_path1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0])
        # MLP (pre-norm)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
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

        # Encoder stack
        self.blocks = nn.ModuleList(
            [
                Encoder(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Returns:
            If num_classes is set: (B, num_classes)
            Else: CLS features (B, D)
        """
        # 1) Patchify & project -> (B, L, D)
        x = self.patch_embed(x)

        # 2) Prepend CLS token -> (B, 1+L, D)
        x = self.class_token(x)

        # 3) Add absolute positional encodings
        #    (PositionalEncoding should slice to current length automatically)
        x = self.positional_encoding(x)

        # 4) Encoder blocks
        for blk in self.blocks:
            x = blk(x)

        # 5) Final norm
        x = self.norm(x)

        # 6) Take CLS token (index 0)
        cls = x[:, 0, :]  # (B, D)

        # 7) Head
        out = self.head(cls)  # (B, num_classes) or identity (B, D)
        return out

def __main__():
    img_folder = '/home/sbangal4/world_in_3d/VGGT/trials/vggt/data/data_co3d/apple/110_13051_23361/images'
    images = load_batches(img_folder, limit=2)
    torch.manual_seed(0)
    model = VisionTransformer(
        img=images,
        batch_size=2,
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