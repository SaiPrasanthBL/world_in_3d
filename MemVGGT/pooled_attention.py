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
from tokenizer import CameraToken, RegisterToken
from memory_bank import MemoryBank
from torch.nn.functional import scaled_dot_product_attention as sdpa

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
        use_memory: bool = True,
        patch_start_idx: None = None
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
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.use_memory = use_memory
        self.head_dim = embed_dim // num_heads
        self.patch_start_idx = patch_start_idx

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

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
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension into (num_heads, head_dim)."""
        B, N, D = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous() # (B, num_heads, N, head_dim)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, num_heads, N, head_dim = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, N, num_heads * head_dim)

    def forward(self, x: torch.Tensor, hw: tuple[int, int, int], memory_bank=None) -> torch.Tensor:
        
        """
        x: (B, S, C) where S = 1 + T*H*W if CLS present, else S = T*H*W
        thw: (T, H, W) for the current token grid
        returns: (B, S, C)  (same sequence length; KV pooling only)
        """

        B, S, C = x.shape
        T, H, W = hw
        has_cls = S == 1 + T * H * W

        x_norm = self.norm1(x)

        # First token is CLS; patches begin at patch_start_idx
        x_cls     = x_norm[:, :1, :]                           # (B,1,C)
        x_spatial = x_norm[:, self.patch_start_idx:, :]        # (B, L, C)

        # if has_cls:
        #     x_cls, x_spatial = x_norm[:, :1, :], x_norm[:, 1:, :]  # (B, 1, C), (B, T*H*W, C)
        # else:
        #     x_cls, x_spatial = None, x_norm  # (B, T*H*W, C)
            
        # Pool key and value tensors if stride > 1
        if self.kv_pool_stride > 1:
            x_spatial_pooled = self._pool_kv(x_spatial, hw, self.kv_pool_stride)  # (B, T*H'*W', C)
        else:
            x_spatial_pooled = x_spatial

        # if has_cls and self.kv_cls_flag:
        #     # Append CLS token to pooled key and value tensors
        #     x_kv = torch.cat((x_cls, x_spatial_pooled), dim=1)  # (B, 1 + T*H'*W', C)
        # else:
        #     x_kv = x_spatial_pooled 

        x_kv = torch.cat([x_cls, x_spatial_pooled], dim=1) if self.kv_cls_flag else x_spatial_pooled

        Q = self.q_proj(x_norm)  # (B, S, C)
        K = self.k_proj(x_kv)
        V = self.v_proj(x_kv)

        # Split heads for multi-head attention
        Qh = self._split_heads(Q)  # (B, num_heads, S, head_dim)
        Kh = self._split_heads(K)  # (B, num_heads, T*H'*W', head_dim)
        Vh = self._split_heads(V)  # (B, num_heads, T*H'*W', head_dim)

        #read from previous memory bank if available
        if self.use_memory and memory_bank is not None:
            K_mem, V_mem = memory_bank.read(K, V)
        else:
            K_mem, V_mem = None, None
        
        if K_mem is not None and V_mem is not None:
            Kmh = self._split_heads(K_mem)  # (B, num_heads, M, head_dim)
            Vmh = self._split_heads(V_mem)
            K_all = torch.cat((Kmh,Kh), dim=2)
            V_all = torch.cat((Vmh,Vh), dim=2)
        else:
            K_all, V_all = Kh, Vh

        # Scaled dot-product attention
        attn_output = sdpa(
            Qh, K_all, V_all, dropout_p=self.attn.dropout if hasattr(self.attn, 'dropout') else 0.0, is_causal=False
        )
        attn_output = self._merge_heads(attn_output)  # (B, S, C)
        # Apply output projection
        attn_output = self.out_proj(attn_output)
         # (B, T*H'*W', C)
        # Multi-head self-attention (pre-norm)
        #x_attn, _ = self.attn(x_norm, x_kv, x_kv, need_weights=True)
        x = x + self.drop_path1(attn_output)  # Residual connection
        
        # MLP (pre-norm)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        # ===== WRITE memory AFTER attention; base shape (B, Np, D) =====
        if self.use_memory and (memory_bank is not None):
            with torch.no_grad():
                Ks = self.k_proj(x_spatial_pooled)   # (B, Np, D)  pooled spatial only
                Vs = self.v_proj(x_spatial_pooled)   # (B, Np, D)
                memory_bank.write(Ks, Vs)

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

        self.n_cam = 1
        self.n_reg = 4
        self.patch_start_idx = 1 + self.n_cam + self.n_reg

        self.camera_token = CameraToken(embed_dim)
        self.register_token = RegisterToken(n_reg=self.n_reg, embed_dim=embed_dim)
        # +1 for CLS token
        self.positional_encoding = PositionalEncoding(embed_dim, num_patches=self.patch_embed.num_patches)

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
                    kv_cls_flag = True,
                    use_memory=True,  # Enable memory bank usage
                    patch_start_idx=self.patch_start_idx
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
        self.memory_bank = MemoryBank(max_tokens=500, dtype=torch.float16)
        

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
        # 1.5) Prepend Camera tokens -> (B, 2+L, D)
        x = self.camera_token(x)  # (B, 2+L, D)
        # 1.75) Prepend Register tokens -> (B, 2+2+L, D)
        x = self.register_token(x)  # (B, 2+2+L
        # 2) Prepend CLS token -> (B, 1+L, D)
        x = self.class_token(x)
        # 3) Add absolute positional encodings
        #    (PositionalEncoding should slice to current length automatically)
        x = self.positional_encoding(x, patch_start_idx=self.patch_start_idx)

        # S = x.shape[1]
        # L = S - 1
        # T = 1  # Assuming single frame input; can be generalized later
        # H = W = int(math.sqrt(L))
        # hw = (T, H, W)  # Current token grid size

        # 4) Encoder blocks
        for blk in self.blocks:
            x = blk(x, hw, memory_bank = self.memory_bank)

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
    model.memory_bank = MemoryBank(max_tokens=40, dtype=torch.float16)
    logits = model(images)
    print(logits.shape)  # Should print (2, 1000)
    print(logits)
    print(logits.dtype)

if __name__ == "__main__":
    __main__()