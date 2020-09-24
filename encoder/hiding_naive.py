import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
import util
from encoder.prep_naive import PrepNetwork_Naive

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class Hiding_naive(nn.Module):
    def __init__(self,config=GlobalConfig()):
        super(Hiding_naive, self).__init__()
        self.config = config
        # Prep
        self.Prepare = PrepNetwork_Naive(config)

        # Hiding
        self.initialH3 = nn.Sequential(
            DoubleConv(150 + 3, 50, mode=1),
            DoubleConv(50, 50, mode=0),
        )
        self.initialH4 = nn.Sequential(
            DoubleConv(150 + 3, 50, mode=1),
            DoubleConv(50, 50, mode=0),
        )
        self.initialH5 = nn.Sequential(
            DoubleConv(150 + 3, 50, mode=1),
            DoubleConv(50, 50, mode=0),
        )
        self.finalH3 = nn.Sequential(
            DoubleConv(150, 50, mode=1),
            DoubleConv(50, 50, mode=0),
        )
        self.finalH4 = nn.Sequential(
            DoubleConv(150, 50, mode=1),
            DoubleConv(50, 50, mode=0),
        )
        self.finalH5 = nn.Sequential(
            DoubleConv(150, 50, mode=1),
            DoubleConv(50, 50, mode=0),
        )
        self.finalH = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))

    def forward(self, Cover, Another):
        prep = self.Prepare(Another)
        p = torch.cat((prep, Cover), 1)
        h1 = self.initialH3(p)
        h2 = self.initialH4(p)
        h3 = self.initialH5(p)
        h1_cat = torch.cat((h1, h2, h3), 1)
        h4 = self.finalH3(h1_cat)
        h5 = self.finalH4(h1_cat)
        h6 = self.finalH5(h1_cat)
        h2_cat = torch.cat((h4, h5, h6), 1)
        out = self.finalH(h2_cat)
        return out
