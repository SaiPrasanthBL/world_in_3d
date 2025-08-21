# Patch Embedding: Images -> Tensor
# 
# This module takes an image input and converts it into a tensor representation suitable for further processing in a neural network.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv

from utils import load_batches

class PatchEmbed(nn.Module):
    def __init__(self, img, batch_size, patch_size, channels, embed_dim):
        super(PatchEmbed, self).__init__()
        self.img = img
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.channels = channels
        self.embed_dim = embed_dim
        
        # Calculate the number of patches
        H, W = img.shape[-2:]
        self.num_patches = (H // patch_size) * (W // patch_size)
        
        # Define the linear layer for embedding patches
        self.proj = nn.Conv2d(in_channels=channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self,  x: torch.Tensor) -> torch.Tensor:

        patch_tokens = self.proj(x)
        patch_h, patch_w = patch_tokens.shape[2], patch_tokens.shape[3]
        L = patch_h * patch_w
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        return patch_tokens

class ClassToken(nn.Module):
    def __init__(self, embed_dim):
        super(ClassToken, self).__init__()
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, patch_start_idx: int) -> torch.Tensor:
        B, S, D = x.shape
        L = S - patch_start_idx
        pe = x.new_zeros(B, S, D)
        pe[:, :1, :] = self.pos_embed[:, :1, :]
        pe[:, patch_start_idx:, :] = self.pos_embed[:, 1:1+L, :]
        x = x + pe
        return x

if __name__ == "__main__":
    img_folder = '/home/sbangal4/world_in_3d/VGGT/trials/vggt/data/data_co3d/apple/110_13051_23361/images'
    images = load_batches(img_folder, limit=2)
    torch.manual_seed(0)
    pe = PatchEmbed(images, batch_size=2, patch_size=16, channels=3, embed_dim=768)
    tokens = pe(images)
    print(tokens.shape)  # Should print the shape of the patch tokens
    print(tokens)
    print(tokens.dtype)
    ct = ClassToken(embed_dim=768)
    full_tokens = ct(tokens)
    print(full_tokens.shape)  # Should print the shape with class token added
    posit = PositionalEncoding(embed_dim=768, num_patches=pe.num_patches)
    pos_tokens = posit(full_tokens)
    print(pos_tokens.shape)
