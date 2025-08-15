import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv

def load_batches(folder, limit=None):
    image_list = []
    for i, f in enumerate(sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])):
        if limit and i >= limit:
            break
        path = os.path.join(folder, f)
        img_bgr = cv.imread(path)
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        x = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float() / 255.0
        image_list.append(x)
    return torch.stack(image_list, dim=0)