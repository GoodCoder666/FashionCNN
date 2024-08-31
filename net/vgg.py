import torch
from torch import nn
from functools import partial

__all__ = ['VGG11', 'VGG13', 'VGG16_1', 'VGG16_3', 'VGG19']

def vgg_block(num_convs, in_channels, out_channels, use_1x1conv=False):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    if use_1x1conv:
        layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, conv_arch=None):
        super().__init__()
        conv_blocks = []
        for (num_convs, out_channels, use_1x1conv) in conv_arch:
            conv_blocks.append(vgg_block(num_convs, in_channels, out_channels, use_1x1conv))
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_blocks)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


arch11 = ((1, 64, False), (1, 128, False), (2, 256, False), (2, 512, False), (2, 512, False))
arch13 = ((2, 64, False), (2, 128, False), (2, 256, False), (2, 512, False), (2, 512, False))
arch16_1 = ((2, 64, False), (2, 128, False), (2, 256, True), (2, 512, True), (2, 512, True))
arch16_3 = ((2, 64, False), (2, 128, False), (3, 256, False), (3, 512, False), (3, 512, False))
arch19 = ((2, 64, False), (2, 128, False), (4, 256, False), (4, 512, False), (4, 512, False))

VGG11 = partial(VGG, conv_arch=arch11)
VGG13 = partial(VGG, conv_arch=arch13)
VGG16_1 = partial(VGG, conv_arch=arch16_1)
VGG16_3 = partial(VGG, conv_arch=arch16_3)
VGG19 = partial(VGG, conv_arch=arch19)