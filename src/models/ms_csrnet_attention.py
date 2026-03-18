import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# -----------------------------
# Channel Attention (Lightweight)
# -----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


# -----------------------------
# Multi-Scale Block
# -----------------------------
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleBlock, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, 128, 3, padding=1, dilation=1)
        self.branch2 = nn.Conv2d(in_channels, 128, 3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(in_channels, 128, 3, padding=4, dilation=4)

        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, x):

        b1 = self.relu(self.branch1(x))
        b2 = self.relu(self.branch2(x))
        b3 = self.relu(self.branch3(x))

        w = torch.softmax(self.weights, dim=0)

        out = w[0]*b1 + w[1]*b2 + w[2]*b3

        out = self.relu(self.bn(out))

        return out


# -----------------------------
# MS-CSRNet + Attention
# -----------------------------
class MSCSRNet_Attention(nn.Module):
    def __init__(self, pretrained=True):
        super(MSCSRNet_Attention, self).__init__()

        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())

        # Frontend
        self.frontend = nn.Sequential(*features[:-2])

        # Multi-scale
        self.ms_block = MultiScaleBlock(512)

        # ✅ Attention added HERE
        self.attention = ChannelAttention(128)

        # Backend
        self.backend = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )

        # Regressor
        self.regressor = nn.Conv2d(64, 1, 1)

        self._initialize_weights()

    def forward(self, x):

        x = self.frontend(x)

        x = self.ms_block(x)

        x = self.attention(x)   # 🔥 EXACT PLACE

        x = self.backend(x)

        x = self.regressor(x)

        x = F.interpolate(
            x,
            scale_factor=16,
            mode='bilinear',
            align_corners=False
        )

        return x

    def _initialize_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.regressor.weight, std=0.01)
        nn.init.constant_(self.regressor.bias, 0)