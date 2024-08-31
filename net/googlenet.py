import torch
from torch import nn


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.BatchNorm2d(c1),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.BatchNorm2d(c2[0]), nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c2[1]), nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.BatchNorm2d(c3[0]), nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(c3[1]), nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.BatchNorm2d(c4), nn.ReLU()
        )

    def forward(self, x):
        return torch.cat((self.p1(x), self.p2(x), self.p3(x), self.p4(x)), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=5),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), 
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128)
        )
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.fc(self.b5(self.b4(self.b3(self.b2(self.b1(x))))))