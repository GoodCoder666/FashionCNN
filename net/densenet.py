import torch
from torch import nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_channels, num_convs):
        super().__init__()
        self.net = nn.Sequential(
            *(conv_block(in_channels + num_channels * i, num_channels) for i in range(num_convs))
        )

    def forward(self, X):
        for block in self.net:
            Y = block.forward(X)
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

class DenseNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        num_channels, growth_rate = 64, 32
        dense_convs = [4, 4, 4, 4]
        blocks = []
        for i, num_convs in enumerate(dense_convs):
            blocks.append(DenseBlock(num_channels, growth_rate, num_convs))
            num_channels += growth_rate * num_convs
            if i != len(dense_convs) - 1:
                blocks.append(transition_block(num_channels, num_channels // 2))
                num_channels //= 2
        self.b2 = nn.Sequential(
            *blocks,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_channels, num_classes)
        )

    def forward(self, x):
        return self.b2(self.b1(x))