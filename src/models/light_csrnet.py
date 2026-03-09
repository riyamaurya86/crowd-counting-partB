import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LightCSRNet(nn.Module):
    def __init__(self, pretrained=True):
        super(LightCSRNet, self).__init__()

        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())

        # Use fewer frontend layers
        self.frontend = nn.Sequential(*features[:16])

        # Lightweight backend
        self.backend = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )

        self.regressor = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        x = self.frontend(x)

        x = self.backend(x)

        x = self.regressor(x)

        x = F.interpolate(
            x,
            scale_factor=8,
            mode='bilinear',
            align_corners=False
        )

        return x