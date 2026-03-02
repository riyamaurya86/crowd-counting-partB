import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_count_metrics(pred_density, gt_density):
    """
    Compute MAE, MSE, RMSE based on total count.
    """

    pred_count = pred_density.sum().item()
    gt_count = gt_density.sum().item()

    mae = abs(pred_count - gt_count)
    mse = (pred_count - gt_count) ** 2
    rmse = np.sqrt(mse)

    return mae, mse, rmse


def compute_psnr(pred_density, gt_density):
    """
    Compute PSNR between predicted and GT density maps.
    """

    mse = F.mse_loss(pred_density, gt_density).item()

    if mse == 0:
        return 100

    max_pixel = gt_density.max().item()
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)

    return psnr


def compute_ssim(pred_density, gt_density):
    """
    Compute SSIM between predicted and GT density maps.
    """

    pred = pred_density.squeeze().cpu().numpy()
    gt = gt_density.squeeze().cpu().numpy()

    ssim_value = ssim(
        gt,
        pred,
        data_range=gt.max() - gt.min()
    )

    return ssim_value