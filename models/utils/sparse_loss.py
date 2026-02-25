import torch
import torch.nn.functional as F



def masked_l1_loss_2d(pred: torch.Tensor, 
                   target: torch.Tensor, 
                   lengths: torch.Tensor)-> torch.Tensor:
    """
    Compute L1 loss ignoring time-padding.

    Args:
        pred (Tensor): [B, C, J, T] predicted output
        target (Tensor): [B, C, J, T] target output
        lengths (Tensor): [B] valid lengths in time (T) per sample

    Returns:
        Scalar tensor: average L1 loss over valid elements
    """
    B, C, J, T = pred.shape
    device = pred.device
    time_range = torch.arange(T, device=device).view(1, 1, 1, T)
    # [B, 1, 1, 1] -> broadcasted to [B, 1, 1, T]
    length_mask = (time_range < lengths.view(B, 1, 1, 1)).float()
    l1 = torch.abs(pred - target)
    masked_l1 = l1 * length_mask

    denom = length_mask.sum() * J * C
    return masked_l1.sum() / denom



def masked_smooth_l1_loss_2d(pred, target, lengths, beta=1.0):
    """
    Smooth L1 loss with masking for padded time steps.

    Args:
        pred (Tensor): [B, C, J, T] predicted output
        target (Tensor): [B, C, J, T] ground truth
        lengths (Tensor): [B] valid lengths in time (T)
        beta (float): transition point from L2 to L1 (same as PyTorch SmoothL1Loss)

    Returns:
        Tensor: scalar loss averaged over valid elements
    """
    B, C, J, T = pred.shape
    device = pred.device
    # Create mask: [B, 1, 1, T]
    time_range = torch.arange(T, device=device).view(1, 1, 1, T)
    mask = (time_range < lengths.view(B, 1, 1, 1)).float()

    # Compute element-wise Smooth L1 manually (same as PyTorch with reduction='none')
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)

    # Apply mask
    masked_loss = loss * mask

    # Normalize
    denom = mask.sum() * J * C
    return masked_loss.sum() / denom



def weighted_l1_loss_2d(pred, target, lengths, weights=0.05):
    """
    Compute L1 loss with weighted loss for padded time steps.
    """
    B, C, J, T = pred.shape
    device = pred.device
    time_range = torch.arange(T, device=device).view(1, 1, 1, T)
    # [B, 1, 1, 1] -> broadcasted to [B, 1, 1, T]
    mask = (time_range < lengths.view(B, 1, 1, 1)).float()
    l1 = torch.abs(pred - target)
    unmasked_l1 = l1 * mask
    masked_l1 = l1 * (1 - mask)
    unmasked_l1 = unmasked_l1.sum() / (mask.sum() * J * C)
    masked_l1 = masked_l1.sum() / ((1 - mask).sum() * J * C)
    return unmasked_l1 + masked_l1 * weights


def weighted_l1_loss_1d(pred, target, lengths, weights=0.05):
    """
    Compute L1 loss with weighted loss for padded time steps.
    """
    B, T, C = pred.shape
    device = pred.device
    time_range = torch.arange(T, device=device).view(1, T, 1)
    # [B, 1, 1] -> broadcasted to [B, T 1]
    mask = (time_range < lengths.view(B, 1, 1)).float()
    l1 = torch.abs(pred - target)
    unmasked_l1 = l1 * mask
    masked_l1 = l1 * (1 - mask)
    unmasked_l1 = unmasked_l1.sum() / (mask.sum() *  C)
    masked_l1 = masked_l1.sum() / ((1 - mask).sum()  * C + 1e-4)
    return unmasked_l1 + masked_l1 * weights
    