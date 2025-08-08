import cv2
import numpy as np
import torch
from torchvision import transforms
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import functools
import math
import warnings

def kl_div(p, q, dim=-1, eps=2**(-8)): # Computes a masked, renormalized KL divergence 𝐾𝐿 (𝑝∥𝑞) along dim, 
                                       # ignoring entries where either distribution is “too small,” then sums over that axis.
                                       # Compute KL across axis dim (default: last). Treat any entry < eps (~0.0039) as “too small” and exclude it.
    p_too_small = p < eps
    q_too_small = q < eps
    too_small = p_too_small | q_too_small   # Build masks for tiny probs. Mark positions where either p or q is tiny. Those positions will be ignored in KL to avoid log(0) and huge/inf values.

    p = p.clone()
    q = q.clone()   # Don't mutate caller's tensors. Work on copies.

    p[too_small] = 0
    q[too_small] = 0    # Zero out tiny entries. Remove tiny mass so it doesn’t affect normalization.

    p = p / p.sum(dim, keepdims=True).detach()
    q = q / q.sum(dim, keepdims=True).detach()  # Renormalize each distribution along dim after trimming tiny entries. detach() on the denominator: gradients won’t flow through the normalization constants (sum). That keeps the scale fixed for backprop; you’re optimizing the shape of p and q only. Note: In PyTorch the kwarg is keepdim (singular). If you literally run keepdims, it’ll error on most versions.

    p[too_small] = 1
    q[too_small] = 1    # Set masked entries to 1 so log() is well-defined (log(1)=0). These won’t contribute anyway (next lines zero them out).

    els = p * (p.log() - q.log())   # Standard KL elementwise operation: 𝑝⋅(log𝑝−log𝑞)
    els[too_small] = 0  # Ignore contributions where either original prob was tiny.
   
    kl_div = els.sum(dim)
    return kl_div   # Aggregate along dim. Shape is the input shape with dim removed (e.g., for [B, C] returns [B]).

def normalized_entropy(p, dim=-1, eps=2**(-8)):

    H_max = math.log2(p.shape[dim]) # Compute the maximum possible entropy. Maximum entropy occurs when the distribution is uniform: pi=1/N, and Hmax=log2(N), where 𝑁 is the number of categories. Using log2 means the result is in bits.

    # x log2 (x) -> 0 . Therefore, we can set log2 (x) to 0 if x is small enough.
    # This should ensure numerical stability.
    p_too_small = p < eps
    p = p.clone()
    p[p_too_small] = 1
    
    plp = torch.log2(p) * p # This is the term used in Shannon entropy

    plp[p_too_small] = 0

    # This is the formula for the normalised entropy
    entropy = -plp.sum(dim) / H_max # Standard Shannon entropy in bits.Dividing by Hmax normalizes it so the result is in [0, 1]: 1 → perfectly uniform distribution, 0 → completely certain (one category has prob 1, rest 0)
    return entropy
        


