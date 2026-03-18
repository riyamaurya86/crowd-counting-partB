import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleBlock, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, 256, 3, padding=1, dilation=1)
        self.branch2 = nn.Conv2d(in_channels, 256, 3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(in_channels, 256, 3, padding=4, dilation=4)

        self.relu = nn.ReLU(inplace=True)

        # Learnable weights
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, x):

        b1 = self.relu(self.branch1(x))
        b2 = self.relu(self.branch2(x))
        b3 = self.relu(self.branch3(x))

        w = torch.softmax(self.weights, dim=0)

        out = w[0]*b1 + w[1]*b2 + w[2]*b3

        return out


class CSRNet_MultiScale(nn.Module):
    def __init__(self, pretrained=True):
        super(CSRNet_MultiScale, self).__init__()

        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())

        self.frontend = nn.Sequential(*features[:-2])

        # Multi-scale block
        self.ms_block = MultiScaleBlock(512)

        # Restore CSRNet backend
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

        x = self.ms_block(x)

        x = self.backend(x)

        x = self.regressor(x)

        x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)

        return x