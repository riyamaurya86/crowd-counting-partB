import os
import random
import h5py
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ShanghaiPartBDataset(Dataset):
    """
    ShanghaiTech Part B Dataset Loader
    Supports:
        - Random crop (train mode)
        - Full image (test mode)
        - Pre-generated h5 density maps
    """

    def __init__(self, root_path, mode="train", crop_size=256):
        """
        Args:
            root_path (str): Path to part_B folder
            mode (str): 'train' or 'test'
            crop_size (int): Random crop size (train only)
        """

        self.root_path = root_path
        self.mode = mode
        self.crop_size = crop_size

        self.image_dir = os.path.join(root_path, f"{mode}_data", "images")
        self.density_dir = os.path.join(root_path, f"{mode}_data", "ground-truth-h5")

        self.image_filenames = [
            fname for fname in os.listdir(self.image_dir)
            if fname.endswith(".jpg")
        ]

        self.image_filenames.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Load density map
        density_name = img_name.replace(".jpg", ".h5")
        density_path = os.path.join(self.density_dir, density_name)

        with h5py.File(density_path, 'r') as hf:
            density = np.array(hf['density'])

        if self.mode == "train":
            image, density = self.random_crop(image, density)

        # Convert image to tensor
        image = self.transform(image)

        # Convert density to tensor
        density = torch.from_numpy(density).unsqueeze(0).float()

        return {
            "image": image,
            "density": density,
            "count": density.sum().item()
        }

    def random_crop(self, image, density):
        h, w, _ = image.shape
        crop_h = min(self.crop_size, h)
        crop_w = min(self.crop_size, w)

        if h == crop_h and w == crop_w:
            return image, density

        x = random.randint(0, h - crop_h)
        y = random.randint(0, w - crop_w)

        image = image[x:x + crop_h, y:y + crop_w]
        density = density[x:x + crop_h, y:y + crop_w]

        return image, density