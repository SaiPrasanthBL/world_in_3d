import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np  
import math

class MemoryBank:
    def __init__(self, max_tokens: int, dtype = torch.float16):
        self.max_tokens = max_tokens
        self.dtype = dtype
        self.K = None
        self.V = None

    @torch.no_grad()
    def write(self, K: torch.Tensor, V: torch.Tensor):
        if self.K is None:
            self.K = K.detach().to(self.dtype)
            self.V = V.detach().to(self.dtype)
        else:
            self.K = torch.cat((self.K, K.detach().to(self.dtype)), dim=1)
            self.V = torch.cat((self.V, V.detach().to(self.dtype)), dim=1)
        
        if self.K.shape[1] > self.max_tokens:
            excess = self.K.shape[1] - self.max_tokens
            self.K = self.K[:,excess:,:]
            self.V = self.V[:,excess:,:]

    @torch.no_grad()
    def read(self, K: torch.Tensor, V: torch.Tensor):
        return self.K, self.V

    @torch.no_grad()
    def reset(self):
        self.K = None
        self.V = None