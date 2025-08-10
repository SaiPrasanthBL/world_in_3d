import math
import torch
import torch.nn.functional as F

from scenedino.common.util import kl_div, normalized_entropy
from scenedino.models.prediction_heads.layers import ssim, geo

def compute_l1ssim(
    img0: torch.Tensor, img1: torch.Tensor, mask: torch.Tensor | None = None    # Defines a function that takes two image batches (img0 = prediction, img1 = ground truth), both shaped (B, C, H, W). An optional mask (B, H, W) can be passed (not actually used below).
) -> torch.Tensor:  ## (img0 == pred, img1 == GT)
    """Calculate the L1-SSIM error between two images. Use a mask if provided to ignore certain pixels.

    Args:
        img0 (torch.Tensor): torch.Tensor of shape (B, c, h, w) containing the predicted images.
        img1 (torch.Tensor): torch.Tensor of shape (B, c, h, w) containing the ground truth images.
        mask (torch.Tensor | None, optional): torch.Tensor of shape (B, h, w). Defaults to None.

    Returns:
        torch.Tensor: per patch error of shape (B, h, w)
    """
    errors = 0.85 * torch.mean(
        ssim(img0, img1, pad_reflection=False, gaussian_average=True, comp_mode=True),
        dim=1,
    ) + 0.15 * torch.mean(torch.abs(img0 - img1), dim=1)
    # checking if a mask is provided. If a mask is provided, it is returned along with the errors. Otherwise, only the errors are returned.
    # if mask is not None:
    #     return (
    #         errors,
    #         mask,
    #     )
    return errors  # (B, h, w)

    # Calls ssim(...) on the two images: pad_reflection=False: uses zero padding at borders inside SSIM.
    # gaussian_average=True: SSIM uses Gaussian-weighted local stats.
    # comp_mode=True: returns an SSIM-based error (roughly (1 - SSIM)/2), not the raw SSIM index.
    # ssim(...) returns a per-channel error map (B, C, H, W).
    # torch.mean(..., dim=1) averages over channels → (B, H, W).
    # Multiplies that SSIM error by 0.85 (weighting). Adds 0.15 * mean(|img0 - img1|, dim=1): 
    # torch.abs(img0 - img1) is per-channel L1 (B, C, H, W). torch.mean(..., dim=1) averages L1 across channels → (B, H, W).
    # Net effect: composite error = 0.85·SSIM_loss + 0.15·L1_loss, per pixel. Common photometric mix.

def compute_normalized_l1(flow0: torch.Tensor, flow1: torch.Tensor) -> torch.Tensor:    # Defines a helper that compares two flow fields (usually optical flow). Typical shape is (B, C, H, W) where C=2 for (u, v).

    errors = (flow0 - flow1).abs() / (flow0.detach().norm(dim=1, keepdim=True) + 1e-4)

    return errors

    # Breakdown:
    # (flow0 - flow1) - Elementwise residual between the two flows → shape (B, C, H, W).
    # .abs() - Absolute value per element → still (B, C, H, W).
    # This is an L1-style per-component error (per channel), not a vector norm.
    # flow0.detach() - Stops gradients flowing through the denominator. The scale you divide by is treated as a constant:
    # Prevents the model from “cheating” by changing flow0 just to alter the normalization scale.
    # Improves numerical stability (no gradient coupling via the norm).
    # .norm(dim=1, keepdim=True) - L2 norm across the channel dimension at each pixel (e.g., sqrt(u^2 + v^2) for 2D flow).
    # Result shape: (B, 1, H, W) so it broadcasts when dividing the (B, C, H, W) numerator.
    # + 1e-4 - Epsilon to avoid divide-by-zero when the flow magnitude is tiny. Sets a small floor for the scale.

def compute_edge_aware_smoothness(
    gt_img: torch.Tensor, input: torch.Tensor, mask: torch.Tensor | None = None, temperature: int = 1
) -> torch.Tensor:
    """Compute the edge aware smoothness loss of the depth prediction based on the gradient of the original image.

    Args:
        gt_img (torch.Tensor): ground truth images of shape (B, c, h, w)
        input (torch.Tensor): predicted tensor of shape (B, c, h, w)
        mask (torch.Tensor | None, optional): Not used yet. Defaults to None.

    Returns:
        torch.Tensor: per pixel edge aware smoothness loss of shape (B, h, w)
    """
    # Computes edge-aware smoothness loss for depth prediction based on gradients of the original image.
    # Goal: penalize spatial roughness in input (e.g., depth or flow) less at places where the image has edges (so the prediction can change at real image edges).
    # gt_img: reference image(s), shape (B, C, H, W).
    # input: prediction field to be smoothed, shape (B, C, H, W) (often C=1 for depth).
    # mask: unused for now. 
    # temperature: controls how strongly image edges downweight the smoothness penalty.
    _, _, h, w = gt_img.shape

    # TODO: check whether interpolation is necessary
    # gt_img = F.interpolate(gt_img, (h, w))

    input_dx = torch.mean(
        torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:]), 1, keepdim=True
    )  # (B, 1, h, w-1)
    # Horizontal finite difference (along width): I(x,y) - I(x,y+1). Takes absolute value (magnitude), then mean over channels (dim=1) so it becomes a single-channel edge magnitude. Result is narrower by 1 column: (B, 1, H, W-1).

    input_dy = torch.mean(
        torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]), 1, keepdim=True
    )  # (B, 1, h-1, w)
    # Vertical finite difference (along height): I(x,y) - I(x+1,y). Same channel-mean. Result is shorter by 1 row: (B, 1, H-1, W).

    i_dx = torch.mean(
        torch.abs(gt_img[:, :, :, :-1] - gt_img[:, :, :, 1:]), 1, keepdim=True
    )  # (B, 1, h, w-1)
    # Horizontal image gradient magnitude (channel-averaged). Same spatial size as input_dx.

    i_dy = torch.mean(
        torch.abs(gt_img[:, :, :-1, :] - gt_img[:, :, 1:, :]), 1, keepdim=True
    )  # (B, 1, h-1, w)
    # Vertical image gradient magnitude. Same spatial size as input_dy.

    input_dx *= torch.exp(-temperature * i_dx)  # (B, 1, h, w-1)
    input_dy *= torch.exp(-temperature * i_dy)  # (B, 1, h-1, w)
    # Where the image gradient is large (strong edge), exp(-T * i_d*) is small, so the smoothness penalty on input differences is suppressed → allows sharp changes at real edges.
    # temperature scales how aggressively you downweight near edges:
    # larger temperature → stronger suppression at edges.

    errors = F.pad(input_dx, pad=(0, 1), mode="constant", value=0) + F.pad(
        input_dy, pad=(0, 0, 0, 1), mode="constant", value=0
    )  # (B, 1, h, w)
    # F.pad pad order is (left, right, top, bottom).
    # For input_dx (shape (B,1,H,W-1)), pad=(0,1) adds 1 column on the right, yielding (B,1,H,W).
    # For input_dy (shape (B,1,H-1,W)), pad=(0,0,0,1) adds 1 row at the bottom, yielding (B,1,H,W).
    # Sum the two (horizontal + vertical) to get a per-pixel smoothness penalty.
    return errors[:, 0, :, :]  # (B, h, w)
    # Remove the singleton channel → final per-pixel loss map (B, H, W).

    # Scale matters: If gt_img is in [0, 255], image gradients are big; consider normalizing to [0, 1] or adjust temperature.
    # Stability: Consider torch.exp(-T * i_d*) clamping or using torch.exp(-T * i_d*.clamp_max(k)) if you see underflow to ~0 for large edges.
    # Alternative filters: Sobel/Scharr filters give smoother gradients than simple finite differences.
    # Channel handling: Using mean over channels is fine; for RGB you might also try converting to luminance before differencing.
    # Mask: If you plan to use mask, apply it when aggregating (e.g., weighted average of errors).

def compute_3d_smoothness(
    feature_sample: torch.Tensor, sigma_sample: torch.Tensor
) -> torch.Tensor:
    # Defines a function that takes: feature_sample: a tensor of features sampled along some third axis (often samples along a ray or along 3D neighbors). 
    # Common shape is (B, C, S) or (B, N, S), where S is the number of samples you want to be “smooth” across.
    # sigma_sample: an extra tensor (e.g., densities/uncertainties) that is not used in this implementation.
    # Returns the variance of feature_sample along axis 2 (the third dimension), i.e., across the S samples.
    # If feature_sample has shape (B, C, S), the output shape is (B, C).
    # By default, torch.var(..., dim=...) uses the unbiased estimator (Bessel’s correction). In newer PyTorch this is correction=1 by default.
    # Consequence: if S == 1, this returns NaN (division by S-1 = 0). If you want a well-defined “population” variance, use: torch.var(feature_sample, dim=2, correction=0)  # or unbiased=False on older PyTorch.
    # Conceptually, this is a smoothness penalty: low variance across samples ⇒ “smoother” features along that dimension; high variance ⇒ more abrupt changes.

    return torch.var(feature_sample, dim=2)

def compute_occupancy_error(
    teacher_field: torch.Tensor,
    student_field: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the distillation error between the teacher and student density.

    Args:
        teacher_density (torch.Tensor): teacher occpancy map of shape (B)
        student_density (torch.Tensor): student occupancy map of shape (B)
        mask (torch.Tensor | None, optional): Mask indicating bad occpancy values for student or teacher, e.g. invalid occupancies due to out of frustum. Defaults to None.

    Returns:
        torch.Tensor: distillation error of shape (B)
    """

    # Defines a function to measure how close the student occupancy field is to the teacher (a distillation loss). All inputs are PyTorch tensors; mask is optional.

    if mask is not None:
        teacher_field = teacher_field[mask]
        student_field = student_field[mask]
    
    # If a mask is provided, it filters both tensors: 
    #   If mask is shape (B,) (bool), this selects a subset of batch items.
    #   If mask matches the entire tensor shape, it selects all True elements flattened, losing structure.
    #   If mask has mismatched shape, this will error. Make sure the mask indexes the first dimension (batch) unless you intend a full-element mask.

    return torch.nn.MSELoss(reduction="mean")(teacher_field, student_field)  # (1) Computes mean squared error between teacher and student over all remaining elements → returns a single scalar loss.

def depth_regularization(depth: torch.Tensor) -> torch.Tensor:
    # This is a Tikhonov regularization (L2 on gradients) for depth maps. It’s a common smoothness prior in 
    # depth estimation, optical flow, and disparity prediction to encourage locally smooth surfaces.

    """Compute the depth regularization loss.

    Args:
        depth (torch.Tensor): depth map of shape (B, 1, h, w)

    Returns:
        torch.Tensor: depth regularization loss of shape (B)
    """
    # Function definition and docstring. Takes in depth, a tensor representing predicted depth maps. Expected shape: (B, 1, h, w). 
    # B → batch size, 1 → single channel (depth is scalar per pixel), h, w → spatial height and width of the image. Output: a scalar (averaged) loss encouraging smoothness in depth maps.

    depth_grad_x = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    # Computing gradient along the height (vertical direction), depth[:, :, 1:, :] → all depth values except the first row.
    # depth[:, :, :-1, :] → all depth values except the last row. Subtracting them gives finite differences between each pixel and the one directly above it.
    # This is essentially ∂𝐷/∂𝑦 (vertical derivative), but computed discretely.

    depth_grad_y = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    # Computing gradient along the width (horizontal direction), depth[:, :, :, 1:] → all depth values except the first column.
    # depth[:, :, :, :-1] → all depth values except the last column. Subtracting gives finite differences between each pixel and the one directly to its left.
    # This is essentially ∂𝐷/∂𝑥 (horizontal derivative), also computed discretely.

    depth_reg_loss = (depth_grad_x**2).mean() + (depth_grad_y**2).mean()
    # Loss computation, depth_grad_x**2 and depth_grad_y**2 → square the gradients to penalize large changes (smoothness constraint).
    # .mean() → average over all pixels and batch entries for each gradient direction.
    # Add the vertical and horizontal terms together to get the total smoothness penalty.

    return depth_reg_loss
    # Returns a single scalar measuring how smooth the depth map is. 
    # Interpretation: If depth varies smoothly between neighboring pixels → loss is small.
    # If depth changes abruptly → loss increases. This helps remove noisy spikes in depth predictions while preserving large object boundaries (in moderation).




