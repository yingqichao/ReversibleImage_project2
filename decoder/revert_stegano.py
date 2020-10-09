import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
from network.single_conv import SingleConv
from network.pure_upsample import PureUpsampling
from network.single_de_conv import SingleDeConv

class RevertStegano(nn.Module):
    def __init__(self,config=GlobalConfig()):
        super(RevertStegano, self).__init__()
        self.config = config
        # input channel: 3, output channel: 96
        """Features with Kernel Size 7---->channel:64 """
        self.downsample_8 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ELU(inplace=True)
        )
        # 128
        self.downsample_7 = nn.Sequential(
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=2, padding=4),
            # SingleConv(64, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1)
        )

        # 64
        self.downsample_6 = nn.Sequential(
            SingleConv(128, out_channels=128, kernel_size=5, stride=1, dilation=2, padding=4),
            # SingleConv(128, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 32
        self.downsample_5 = nn.Sequential(
            SingleConv(256, out_channels=128, kernel_size=5, stride=1, dilation=2, padding=4),
            # SingleConv(256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # dilated conv
        self.dilate_convs = nn.Sequential(
            SingleConv(384, out_channels=128, kernel_size=3, stride=1, dilation=2, padding=2),
            SingleConv(128, out_channels=128, kernel_size=3, stride=1, dilation=4, padding=4),
            SingleConv(128, out_channels=128, kernel_size=3, stride=1, dilation=8, padding=8)
        )
        # 64
        self.upsample3_3 = nn.Sequential(
            SingleConv(512, out_channels=128, kernel_size=5, stride=1, dilation=2, padding=4),
            # SingleConv(128, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 128
        self.upsample2_3 = nn.Sequential(
            SingleConv(640, out_channels=64, kernel_size=5, stride=1, dilation=2, padding=4),
            # SingleConv(64, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 256
        self.upsample1_3 = nn.Sequential(
            SingleConv(704, out_channels=64, kernel_size=5, stride=1, dilation=2, padding=4),
            # SingleConv(32, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1)
        )

        self.finalH1 = nn.Sequential(
            SingleConv(64, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.Conv2d(64, 3, kernel_size=1, padding=0),
        )
        # self.finalH2 = nn.Sequential(
        #     SingleConv(67, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1),
        #     # SingleConv(32, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1),
        #     nn.Conv2d(32, 3, kernel_size=1, padding=0),
        #     nn.Tanh()
        # )

    def forward(self, p):
        # Features with Kernel Size 7
        down8 = self.downsample_8(p)
        down7 = self.downsample_7(down8)
        down7_cat = torch.cat((down8, down7), 1)
        down6 = self.downsample_6(down7_cat)
        down6_cat = torch.cat((down8, down7, down6), 1)
        down5 = self.downsample_5(down6_cat)
        down5_cat = torch.cat((down8, down7, down6, down5), 1)
        dilate = self.dilate_convs(down5_cat)
        dilate_cat = torch.cat((down8, down7, down6, down5, dilate), 1)
        up3 = self.upsample3_3(dilate_cat)
        up3_cat = torch.cat((down8, down7, down6, down5, dilate, up3), 1)
        up2 = self.upsample2_3(up3_cat)
        up2_cat = torch.cat((down8, down7, down6, down5, dilate, up3, up2), 1)
        up1 = self.upsample1_3(up2_cat)
        # up1_cat = torch.cat((down8, down7, down6, down5, dilate, up3, up2, up1), 1)
        up0 = self.finalH1(up1)
        out = p + up0
        return out
