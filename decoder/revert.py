import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
import util
from network.single_conv import SingleConv
from network.pure_upsample import PureUpsampling

class Revert(nn.Module):
    def __init__(self,input_channel=96, config=GlobalConfig()):
        super(Revert, self).__init__()
        self.config = config
        # input channel: 96+3, output channel: 3
        self.retrieve1 = DoubleConv(input_channel + 3, 32, mode=0)
        self.retrieve2 = DoubleConv(input_channel + 3, 32, mode=1)
        self.retrieve3 = DoubleConv(input_channel + 3, 32, mode=2)

        self.finalH = nn.Sequential(
            nn.Conv2d(96, 3, kernel_size=1, padding=0))


    def forward(self, p, ori_image):
        # Features with Kernel Size 7
        mix = torch.cat((p, ori_image), 1)
        mid1 = self.retrieve1(mix)
        mid2 = self.retrieve2(mix)
        mid3 = self.retrieve2(mix)
        out = torch.cat((mid1,mid2,mid3), 1)
        recover = self.finalH(out)
        return recover
