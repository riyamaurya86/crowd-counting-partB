import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CSRNet(nn.Module):
    """
    CSRNet implementation:
    - VGG16 frontend (pretrained)
    - Dilated convolution backend
    - 1x1 density regressor
    """

    def __init__(self, pretrained=True):
        super(CSRNet, self).__init__()

        # --------------------------
        # Frontend (VGG16)
        # --------------------------
        vgg = models.vgg16(pretrained=pretrained)

        # Extract first 23 layers (up to conv5_3)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        # --------------------------
        # Backend (Dilated Convs)
        # --------------------------
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        # --------------------------
        # Density Regressor
        # --------------------------
        self.regressor = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.regressor(x)

        # Upsample to input resolution (8x downsampled → restore)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)

        return x

    def _initialize_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.regressor.weight, std=0.01)
        nn.init.constant_(self.regressor.bias, 0)