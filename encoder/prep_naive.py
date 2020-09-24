import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
import util

class PrepNetwork_Naive(nn.Module):
    def __init__(self,config=GlobalConfig()):
        super(PrepNetwork_Naive, self).__init__()
        self.config = config
        # Prep
        self.initialP3 = nn.Sequential(
            DoubleConv(3, 50, mode=1),  # 3*3
            DoubleConv(50, 50, mode=0),
        )
        self.initialP4 = nn.Sequential(
            DoubleConv(3, 50, mode=1),  # 3*3
            DoubleConv(50, 50, mode=0),
        )
        self.initialP5 = nn.Sequential(
            DoubleConv(3, 50, mode=1),  # 3*3
            DoubleConv(50, 50, mode=0),
        )
        self.finalP3 = nn.Sequential(
            DoubleConv(150, 50, mode=1),
            DoubleConv(50, 50, mode=0),
        )
        self.finalP4 = nn.Sequential(
            DoubleConv(150, 50, mode=1),
            DoubleConv(50, 50, mode=0),
        )
        self.finalP5 = nn.Sequential(
            DoubleConv(150, 50, mode=1),
            DoubleConv(50, 50, mode=0),
        )



    def forward(self, p):
        p1 = self.initialP3(p)
        p2 = self.initialP4(p)
        p3 = self.initialP5(p)
        p1_cat = torch.cat((p1, p2, p3), 1)
        p4 = self.finalP3(p1_cat)
        p5 = self.finalP4(p1_cat)
        p6 = self.finalP5(p1_cat)
        mid = torch.cat((p4, p5, p6), 1)

        return mid
