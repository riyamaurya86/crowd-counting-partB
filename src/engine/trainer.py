import os
import torch
from tqdm import tqdm

from src.metrics.metrics import (
    compute_count_metrics,
    compute_psnr,
    compute_ssim
)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)
        gt_density = batch["density"].to(device)

        optimizer.zero_grad()

        pred_density = model(images)

        loss = criterion(pred_density, gt_density)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss


def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch["image"].to(device)
            gt_density = batch["density"].to(device)

            pred_density = model(images)

            loss = criterion(pred_density, gt_density)
            total_loss += loss.item()

            # Compute metrics per image
            for i in range(images.size(0)):
                pred = pred_density[i]
                gt = gt_density[i]

                mae, mse, _ = compute_count_metrics(pred, gt)
                psnr = compute_psnr(pred, gt)
                ssim = compute_ssim(pred, gt)

                total_mae += mae
                total_mse += mse
                total_psnr += psnr
                total_ssim += ssim

    num_samples = len(dataloader.dataset)

    results = {
        "loss": total_loss / len(dataloader),
        "mae": total_mae / num_samples,
        "mse": total_mse / num_samples,
        "rmse": (total_mse / num_samples) ** 0.5,
        "psnr": total_psnr / num_samples,
        "ssim": total_ssim / num_samples,
    }

    return results


def save_checkpoint(model, optimizer, epoch, best_mae, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_mae": best_mae
    }, path)