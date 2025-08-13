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

def alpha_regularization(
        alphas: torch.Tensor, invalids: torch
) -> torch.Tensor:
    """Compute the alpha regularization loss.

    Args:
        alphas (torch.Tensor): alpha map of shape (B, 1, h, w)
        invalids (torch.Tensor): mask indicating invalid alphas of shape (B, 1, h, w)

    Returns:
        torch.Tensor: alpha regularization loss of shape (B)
    """
    # Computes a regularization loss on alpha maps to encourage smoothness and penalize noise in a ray-marching or volumetric rendering context.

    # Function definition:
    # alphas: tensor of alpha values (opacity/transparency) along sampled points in a ray (shape (B, 1, h, w) in the docstring, but here it seems last dim indexes ray samples).
    # invalids: optional mask (same size except last dim is 1) indicating invalid alpha entries (e.g., out of frustum, background, masked regions).

    alpha_reg_fraction = 1/8
    alpha_reg_reduction = "ray"

    # Hardcoded params:
    # alpha_reg_fraction: fraction of the maximum possible alpha sum used as a minimum allowed alpha mass per ray. Here 1/8 means at least 12.5% of the maximum alpha sum is expected.
    # alpha_reg_reduction: determines how to reduce the alpha penalty — "ray" applies per-ray, "slice" aggregates across slices.

    n_smps = alphas.shape[-1]   # Number of samples along each ray (n_smps = samples per pixel in volume rendering)

    alpha_sum = alphas[..., :-1].sum(-1)    # Sum of alpha values along the ray, excluding the very last sample (:-1).
                                            # This represents total "opacity" collected from front to back before the ray terminates.

    min_cap = torch.ones_like(alpha_sum) * (n_smps * alpha_reg_fraction)

    # min_cap = a per-ray minimum alpha sum threshold.
    # Computed as number_of_samples * alpha_reg_fraction.
    # This acts like a floor: if a ray’s total alpha is below this, the loss will encourage increasing it.

    if invalids is not None:
        alpha_sum = alpha_sum * (1 - invalids.squeeze(-1).to(torch.float32))
        min_cap = min_cap * (1 - invalids.squeeze(-1).to(torch.float32))

    # If a mask of invalid rays/samples is provided:
    # Zeros out alpha_sum and min_cap for those rays so they don’t contribute to the loss.
    # .squeeze(-1) removes last singleton dim (needed for broadcasting).

    match alpha_reg_reduction:
        case "ray":
            alpha_reg_loss = (alpha_sum - min_cap).clamp_min(0)
        case "slice":
            alpha_reg_loss = (alpha_sum.sum(dim=-1) - min_cap.sum(dim=-1)).clamp_min(
                0
            ) / alpha_sum.shape[-1]
        case _:
            raise ValueError(f"Invalid alpha_reg_reduction: {alpha_reg_reduction}")

    return alpha_reg_loss 

    # Returns the computed alpha regularization loss.
    # Shape depends on reduction mode:
    # "ray" → (B, H, W) per-ray values.
    # "slice" → (B, H) or similar after aggregation.

    # Density control regularizer for NeRF-like rendering:
    # Prevents rays from accumulating too little or too much opacity.
    # Helps avoid “floating transparency” artifacts and encourages compact, bounded surfaces.
    # Invalids mask ensures we don’t penalize in irrelevant areas.

def surfaceness_regularization(
    alphas: torch.Tensor, invalids: torch.Tensor | None = None
) -> torch.Tensor:
    
    # It’s a "surfaceness" prior — encouraging alpha values to be close to either 0 (transparent) or 1 (fully opaque), rather than being fuzzy in between.
    
    # Takes:
    # alphas: opacity values per sample along a ray (B, ..., n_samples).
    # invalids: optional mask for excluding certain samples from the loss.

    p = -torch.log(torch.exp(-alphas.abs()) + torch.exp(-(1 - alphas).abs()))   # Core penalty function.
    p = p.mean(-1)  # Per-ray average penalty.

    if invalids is not None:  # Zeros out penalties for invalid samples/rays.
        p = p * (1 - invalids.squeeze(-1).to(torch.float32))

    surfaceness_reg_loss = p.mean() # Average over all rays in the batch → single scalar loss.
    return surfaceness_reg_loss # Returns scalar loss:
                                # Low when alpha ≈ {0, 1} (sharp surfaces).
                                # High when alpha is mid-range (fuzzy/translucent).

    # Binarization regularizer for alphas in volumetric rendering.
    # It sharpens the surface along the ray by pushing opacity values toward extremes (0 or 1), making geometry look crisp and well-defined instead of smeared.

def depth_smoothness_regularization(depths: torch.Tensor) -> torch.Tensor:
    
    # Smoothness prior for depth maps, similar in spirit to the earlier depth_regularization.
    
    # Input: depths — predicted depth maps.
    # Shape can be (B, 1, H, W) or (B, H, W) depending on context.
    # The ... in indexing means it works even if there are extra leading dimensions.

    depth_smoothness_loss = ((depths[..., :-1, :] - depths[..., 1:, :]) ** 2).mean() + (
        (depths[..., :, :-1] - depths[..., :, 1:]) ** 2
    ).mean()    # Vertical and horizontal smoothness penalties.

    return depth_smoothness_loss # Scalar loss encouraging locally smooth surfaces in the depth map by penalizing pixel-to-pixel jumps.
    # Equivalent to applying an L2 gradient norm to the depth map.
    # If depth is piecewise smooth, the loss is low; if it’s noisy, the loss is high.
    # It’s the same as depth_regularization earlier, just more condensed.

def sdf_eikonal_regularization(sdf: torch.Tensor) -> torch.Tensor:

    # Eikonal loss for signed distance functions (SDFs), commonly used in neural implicit surface learning.
    # Its job is to enforce that the SDF's gradient magnitude is 1 everywhere (a property of true SDFs).

    # sdf: Signed Distance Function values.
    # Likely shaped (B, 1, D, H, W) → batch, channel, depth, height, width.
    # Each voxel contains the signed distance to the surface:
    # Positive → outside the surface.
    # Negative → inside.
    # Zero → on the surface.

    grad_x = sdf[:, :1, :-1, :-1, 1:] - sdf[:, :1, :-1, :-1, :-1]
    grad_y = sdf[:, :1, :-1, 1:, :-1] - sdf[:, :1, :-1, :-1, :-1]
    grad_z = sdf[:, :1, 1:, :-1, :-1] - sdf[:, :1, :-1, :-1, :-1]
    grad = (torch.cat((grad_x, grad_y, grad_z), dim=1) ** 2).sum(dim=1) ** 0.5

    # Finite difference in X (width), Y (height), and Z (depth) directions.
    # Capturing the gradient magnitude of the SDF.

    eikonal_loss = ((grad - 1) ** 2).mean(dim=(1, 2, 3))

    # Eikonal constraint: For a perfect SDF,
    # ∣∇SDF∣ = 1 everywhere.
    # (grad - 1) ** 2 penalizes deviation from 1.
    # .mean(dim=(1, 2, 3)) averages over spatial dimensions, leaving shape (B,) (loss per batch element).

    return eikonal_loss # Per-batch eikonal loss enforcing the SDF property.

    # loss comes from the Eikonal equation:
    # ∥∇f(x)∥=1
    # For an SDF f, the gradient’s magnitude should be exactly 1 at every point.
    # If it’s less than 1 → distances are "squashed".
    # If it’s greater than 1 → distances are "stretched".
    # This regularization keeps the geometry physically consistent.

def weight_entropy_regularization(
    weights: torch.Tensor, invalids: torch.Tensor | None = None
) -> torch.Tensor:
    
    # Entropy regularization for weights, very likely the per-sample contribution weights in a NeRF/volume rendering setting.
    # It penalizes overly “spread out” weights and encourages concentration along rays.

    # Inputs:
    # weights: (B, ..., N) where N is the number of samples along a ray.
    # invalids: mask for ignoring certain rays/samples (not used here, but could be).

    ignore_last = False

    weights = weights.clone()

    if ignore_last:
        weights = weights[..., :-1]
        weights = weights / weights.sum(dim=-1, keepdim=True)

    H_max = math.log2(weights.shape[-1])

    # Maximum possible entropy for N = weights.shape[-1] equally likely samples:
    # H_max = log2(N)
    # This is used later to normalize entropy between 0 and 1.

    # x log2 (x) -> 0 . Therefore, we can set log2 (x) to 0 if x is small enough.
    # This should ensure numerical stability.
    weights_too_small = weights < 2 ** (-16)
    weights[weights_too_small] = 2
    
    wlw = torch.log2(weights) * weights

    wlw[weights_too_small] = 0

    # This is the formula for the normalised entropy
    entropy = -wlw.sum(-1) / H_max
    return entropy

    # Returns normalized entropy:
    # Low entropy (≈ 0) → weights are concentrated (one sample dominates → sharp surface along ray).
    # High entropy (≈ 1) → weights are spread out (blurry/translucent surface).

def max_alpha_regularization(alphas: torch.Tensor, invalids: torch.Tensor | None = None):

    # Alpha regularizer that pushes the maximum alpha along each ray to be close to 1, encouraging strong surface hits in volumetric rendering.

    # alphas: Opacity values along ray samples, shape (B, ..., N).
    # invalids: Optional mask (unused here — could be added for ignoring rays).

    alphas_max = alphas[..., :-1].max(dim=-1)[0]
    alphas_reg = (1 - alphas_max).clamp(0, 1).mean()
    return alphas_reg 

    # Returns a scalar loss:
    # Small if every ray has at least one sample with alpha ≈ 1 (opaque hit).
    # Large if rays are all semi-transparent (max alpha much less than 1).

    # In NeRF-like rendering, alpha values indicate surface opacity per sample.
    # If max alpha along a ray is close to 1 → strong surface detection.
    # This loss encourages rays to commit to a surface rather than staying semi-transparent, improving sharpness and stability.

def max_alpha_inputframe_regularization(alphas: torch.Tensor, ray_info, invalids: torch.Tensor | None = None):
    
    # Variant of the max_alpha_regularization but applied only to a subset of rays determined by ray_info.

    # alphas: (B, ..., N) alpha values along each ray.
    # ray_info: an extra tensor that stores per-ray metadata.
    # invalids: optional mask (not used here).

    mask = ray_info[..., 0] == 0
    alphas_max = alphas.max(dim=-1)[0]
    alphas_reg = ((1 - alphas_max).clamp(0, 1) * mask.to(alphas_max.dtype)).mean()
    return alphas_reg   # Returns scalar loss, encouraging input-frame rays to have at least one sample with alpha ≈ 1.

    # Same goal as max_alpha_regularization: encourage strong surface hits (max alpha near 1).
    # But scope is limited:
    # Instead of all rays, only applies to a subset defined in ray_info[..., 0] == 0.
    # Likely used when supervising reconstruction from specific keyframes while letting novel-view rays be less constrained.

def epipolar_line_regularization(data, rgb_gt, scale):
    rgb = data["coarse"][scale]["rgb"]
    rgb_samps = data["coarse"][scale]["rgb_samps"]

    b, pc, h, w, n_samps, nv, c = rgb_samps.shape

    rgb_gt = data["rgb_gt"].unsqueeze(-2).expand(rgb.shape)

    alphas = data["coarse"][scale]["alphas"]

    # TODO


def density_grid_regularization(density_grid, threshold):
    density_grid = (density_grid.abs() - threshold).clamp_min(0)

    # Attempt to make it more numerically stable
    max_v = density_grid.max().clamp_min(1).detach()

    # print(max_v.item())

    error = (((density_grid / max_v)).mean() * max_v)

    error = torch.nan_to_num(error, 0, 0, 0)

    # Black magic to prevent error massages from anomaly detection when using AMP
    if torch.all(error == 0):
        error = error.detach()

    return error


def kl_prop(weights):
    entropy = normalized_entropy(weights.detach())

    kl_prop = entropy[..., 1:-1, 1:-1] * (entropy[..., 1:-1, 1:-1] - entropy[..., 2:, 1:-1]).clamp_min(0) * kl_div(weights[..., 2:, 1:-1, :].detach(), weights[..., 1:-1, 1:-1, :])
    kl_prop += entropy[..., 1:-1, 1:-1] * (entropy[..., 1:-1, 1:-1] - entropy[..., 0:-2, 1:-1]).clamp_min(0) * kl_div(weights[..., 0:-2, 1:-1, :].detach(), weights[..., 1:-1, 1:-1, :])
    kl_prop += entropy[..., 1:-1, 1:-1] * (entropy[..., 1:-1, 1:-1] - entropy[..., 1:-1, 2:]).clamp_min(0) * kl_div(weights[..., 1:-1, 2:, :].detach(), weights[..., 1:-1, 1:-1, :])
    kl_prop += entropy[..., 1:-1, 1:-1] * (entropy[..., 1:-1, 1:-1] - entropy[..., 1:-1, 0:-2]).clamp_min(0) * kl_div(weights[..., 1:-1, :-2, :].detach(), weights[..., 1:-1, 1:-1, :])

    return kl_prop.mean()


def alpha_consistency(alphas, invalids, consistency_policy):
    invalids = torch.all(invalids < .5, dim=-1)

    if consistency_policy == "max":
        target = torch.max(alphas, dim=-1, keepdim=True)[0].detach()
    elif consistency_policy == "min":
        target = torch.max(alphas, dim=-1, keepdim=True)[0].detach()
    elif consistency_policy == "median":
        target = torch.median(alphas, dim=-1, keepdim=True)[0].detach()
    elif consistency_policy == "mean":
        target = torch.mean(alphas, dim=-1, keepdim=True).detach()
    else:
        raise NotImplementedError

    diff = (alphas - target).abs().mean(dim=-1)

    invalids = invalids.to(diff.dtype)

    diff = (diff * invalids)

    return diff.mean()


def alpha_consistency_uncert(alphas, invalids, uncert):
    invalids = torch.all(invalids < .5, dim=-1)
    alphas = alphas.detach()
    nf = alphas.shape[-1]

    alphas_median = torch.median(alphas, dim=-1, keepdim=True)[0].detach()

    target = (alphas - alphas_median).abs().mean(dim=-1) * (nf / (nf-1))

    diff = (uncert[..., None] - target).abs()

    invalids = invalids.to(diff.dtype)

    diff = (diff * invalids)

    return diff.mean()


def entropy_based_smoothness(weights, depth, invalids=None):
    entropy = normalized_entropy(weights.detach())

    error_fn = lambda d0, d1: (d0 - d1.detach()).abs()

    if invalids is None:
        invalids = torch.zeros_like(depth)

    # up
    kl_prop_up = entropy[..., :-1, :] * (entropy[..., :-1, :] - entropy[..., 1:, :]).clamp_min(0) * error_fn(depth[..., :-1, :], depth[..., 1:, :]) * (1 - invalids[..., :-1, :])
    # down
    kl_prop_down = entropy[..., 1:, :] * (entropy[..., 1:, :] - entropy[..., :-1, :]).clamp_min(0) * error_fn(depth[..., 1:, :], depth[..., :-1, :]) * (1 - invalids[..., 1:, :])
    # left
    kl_prop_left = entropy[..., :, :-1] * (entropy[..., :, :-1] - entropy[..., :, 1:]).clamp_min(0) * error_fn(depth[..., :, :-1], depth[..., :, 1:]) * (1 - invalids[..., :, :-1])
    # right
    kl_prop_right = entropy[..., :, 1:] * (entropy[..., :, 1:] - entropy[..., :, :-1]).clamp_min(0) * error_fn(depth[..., :, 1:], depth[..., :, :-1]) * (1 - invalids[..., :, 1:])

    kl_prop = kl_prop_up.mean() + kl_prop_down.mean() + kl_prop_left.mean() + kl_prop_right.mean()

    return kl_prop.mean()


def flow_regularization(flow, gt_flow, invalids=None):
    flow_reg = (flow[..., 0, :] - gt_flow).abs().mean(dim=-1, keepdim=True)
    
    if invalids is not None:
        flow_reg = flow_reg * (1 - invalids)

    return flow_reg.mean()

