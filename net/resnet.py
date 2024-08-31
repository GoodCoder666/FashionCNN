import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)), inplace=True)
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y, inplace=True)


def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    block = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block.append(Residual(in_channels, num_channels, use_1x1conv=True, stride=2))
        else:
            block.append(Residual(num_channels, num_channels))
    return nn.Sequential(*block)


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = resnet_block(64, 64, 2, first_block=True)
        self.b3 = resnet_block(64, 128, 2)
        self.b4 = resnet_block(128, 256, 2)
        self.b5 = resnet_block(256, 512, 2)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b5(self.b4(self.b3(self.b2(x))))
        x = self.fc(x)
        return x