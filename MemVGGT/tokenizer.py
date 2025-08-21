import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraToken(nn.Module):
    def __init__(self, embed_dim, n_cam=1):
        super(CameraToken, self).__init__()
        self.embed_dim = embed_dim
        self.n_cam = 1
        self.camera_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        camera_tokens = self.camera_token.expand(batch_size, -1, -1)
        x = torch.cat((camera_tokens, x), dim=1)
        return x

class RegisterToken(nn.Module):
    def __init__(self, n_reg, embed_dim):
        super(RegisterToken, self).__init__()
        self.embed_dim = embed_dim
        self.n_reg = 4
        self.register_token = nn.Parameter(torch.randn(1, n_reg, embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        register_tokens = self.register_token.expand(batch_size, -1, -1)
        x = torch.cat((register_tokens, x), dim=1)
        return x