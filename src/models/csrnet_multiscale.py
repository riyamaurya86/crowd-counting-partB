import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleBlock, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, 128, 3, padding=1, dilation=1)
        self.branch2 = nn.Conv2d(in_channels, 128, 3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(in_channels, 128, 3, padding=4, dilation=4)

        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128*3, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1)
        )

    def forward(self, x):

        b1 = self.relu(self.branch1(x))
        b2 = self.relu(self.branch2(x))
        b3 = self.relu(self.branch3(x))

        concat = torch.cat([b1, b2, b3], dim=1)

        weights = self.attention(concat)      # (B,3,1,1)
        weights = torch.softmax(weights, dim=1)

        w1 = weights[:,0:1,:,:]
        w2 = weights[:,1:2,:,:]
        w3 = weights[:,2:3,:,:]

        out = w1*b1 + w2*b2 + w3*b3

        out = self.relu(self.bn(out))

        return out

class CSRNet_MultiScale(nn.Module):
    def __init__(self, pretrained=True):
        super(CSRNet_MultiScale, self).__init__()

        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())

        self.frontend = nn.Sequential(*features[:-2])

        self.ms_block = MultiScaleBlock(512)

        # Adjusted backend for 128 channels
        self.backend = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )

        self.regressor = nn.Conv2d(64, 1, 1)

        self._initialize_weights()

    def forward(self, x):

        x = self.frontend(x)

        x = self.ms_block(x)

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