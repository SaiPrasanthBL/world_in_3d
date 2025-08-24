import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
#box = np.array([300, 300, 300, 300])

image_path = "/home/sbangal4/world_in_3d/VGGT/trials/vggt/data/vkitti/vkitti/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00000.jpg"
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)
h, w = image_np.shape[:2]
grid_size = 32
points = []
labels = []
for y in range(grid_size // 2, h, grid_size):
    for x in range(grid_size // 2, w, grid_size):
        points.append([x, y])
        labels.append(1)  # assume foreground

points = np.array(points)
labels = np.array(labels)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(point_coords=points, point_labels=labels)

plt.figure(figsize=(10, 10))
plt.imshow(image_np)

for i, mask in enumerate(masks):
    if scores[i] < 0.5: 
        continue
    # color = np.random.rand(3)  # random RGB color
    # mask_rgb = np.zeros_like(image_np, dtype=np.float32)
    # for c in range(3):
    #     mask_rgb[:, :, c] = mask * color[c]
    # plt.imshow(mask, alpha=0.4)  
    mask_bool = mask.astype(bool)
    color = np.random.rand(3)  # RGB
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
    colored_mask[mask_bool, :3] = color  # assign color to mask area
    colored_mask[mask_bool, 3] = 0.5     # alpha
    
    plt.imshow(colored_mask)

plt.axis("off")
plt.tight_layout()
plt.savefig("output.png",dpi=300)
plt.show()
print("Saved visualization to output.png")
print(scores)
print(logits)

#img_path = "/home/sbangal4/world_in_3d/VGGT/trials/vggt/data/vkitti/vkitti/Scene01/15-deg-left/frames/rgb/Camera_0/000000.png"
  # ensures RGB format

#predictor.set_image(image, image_format="RGB")