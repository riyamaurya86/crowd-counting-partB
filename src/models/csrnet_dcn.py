import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import DeformConv2d


class DeformableBlock(nn.Module):
    """
    Deformable Convolution Block:
    - Offset conv
    - DeformConv
    - ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(DeformableBlock, self).__init__()

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * 3 * 3,
            kernel_size=3,
            padding=1
        )

        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        x = self.relu(x)
        return x


class CSRNet_DCN(nn.Module):
    """
    CSRNet with Deformable Convolution backend
    """

    def __init__(self, pretrained=True):
        super(CSRNet_DCN, self).__init__()

        # Frontend (same as CSRNet)
        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())
        self.frontend = nn.Sequential(*features[:-2])  # remove last maxpool

        # Backend (replace dilated conv with deformable blocks)
        self.backend = nn.Sequential(
            DeformableBlock(512, 512),
            DeformableBlock(512, 512),
            DeformableBlock(512, 512),
            DeformableBlock(512, 256),
            DeformableBlock(256, 128),
            DeformableBlock(128, 64),
        )

        self.regressor = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.regressor(x)

        x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)

        return x

    def _initialize_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.regressor.weight, std=0.01)
        nn.init.constant_(self.regressor.bias, 0)