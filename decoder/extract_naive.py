import torch
import torch.nn as nn

from config import GlobalConfig
from network.double_conv import DoubleConv


class Extract_naive(nn.Module):
    def __init__(self, config=GlobalConfig()):
        super(Extract_naive, self).__init__()
        self.config = config
        self.initialR3 = nn.Sequential(
            DoubleConv(3, 50, mode=0),
            DoubleConv(50, 50, mode=0),
        )
        self.initialR4 = nn.Sequential(
            DoubleConv(3, 50, mode=1),
            DoubleConv(50, 50, mode=1),
        )
        self.initialR5 = nn.Sequential(
            DoubleConv(3, 50, mode=2),
            DoubleConv(50, 50, mode=2),
        )
        self.finalR3 = nn.Sequential(
            DoubleConv(150, 50, mode=0),
            DoubleConv(50, 50, mode=0),
        )
        self.finalR4 = nn.Sequential(
            DoubleConv(150, 50, mode=1),
            DoubleConv(50, 50, mode=1),
        )
        self.finalR5 = nn.Sequential(
            DoubleConv(150, 50, mode=2),
            DoubleConv(50, 50, mode=2),
        )
        self.finalR = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))

    def forward(self, r):
        r1 = self.initialR3(r)
        r2 = self.initialR4(r)
        r3 = self.initialR5(r)
        mid = torch.cat((r1, r2, r3), 1)
        r4 = self.finalR3(mid)
        r5 = self.finalR4(mid)
        r6 = self.finalR5(mid)
        mid2 = torch.cat((r4, r5, r6), 1)
        out = self.finalR(mid2)

        return out
