import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_predictions(
    model,
    dataset,
    device,
    save_dir,
    indices=None,
    num_samples=5
):
    """
    Publication-quality visualization:
    - Original Image
    - Ground Truth Density Map
    - Predicted Density Map
    - Count comparison overlay
    """

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    if indices is None: 
        indices = np.random.choice(len(dataset), num_samples, replace=False)

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]

            image = sample["image"].unsqueeze(0).to(device)
            gt_density = sample["density"].to(device)

            pred_density = model(image).squeeze(0)

            # Convert tensors to numpy
            image_np = sample["image"].permute(1, 2, 0).cpu().numpy()
            gt_np = gt_density.squeeze().cpu().numpy()
            pred_np = pred_density.squeeze().cpu().numpy()

            gt_count = gt_np.sum()
            pred_count = pred_np.sum()

            # Normalize density maps for visualization only
            gt_vis = gt_np / (gt_np.max() + 1e-8)
            pred_vis = pred_np / (pred_np.max() + 1e-8)

            fig, axes = plt.subplots(1, 3, figsize=(20, 6))

            # Original image
            axes[0].imshow(image_np)
            axes[0].set_title("Original Image", fontsize=14)
            axes[0].axis("off")

            # Ground Truth
            im1 = axes[1].imshow(gt_vis, cmap="jet")
            axes[1].set_title(f"Ground Truth\nCount: {gt_count:.2f}", fontsize=14)
            axes[1].axis("off")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            # Prediction
            im2 = axes[2].imshow(pred_vis, cmap="jet")
            axes[2].set_title(f"Prediction\nCount: {pred_count:.2f}", fontsize=14)
            axes[2].axis("off")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            plt.tight_layout()

            save_path = os.path.join(save_dir, f"sample_{idx}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

    print(f"Saved visualizations to {save_dir}")