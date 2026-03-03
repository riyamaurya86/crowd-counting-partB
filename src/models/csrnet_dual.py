import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from src.models.dual_attention import DualAttention


class CSRNet_Dual(nn.Module):
    def __init__(self, pretrained=True):
        super(CSRNet_Dual, self).__init__()

        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())
        self.frontend = nn.Sequential(*features[:-2])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        self.attention = DualAttention(64)

        self.regressor = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)

        x = self.attention(x)

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