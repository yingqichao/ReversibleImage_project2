from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
import util
import torch
import torch.nn as nn
from network.single_conv import SingleConv
from network.pure_upsample import PureUpsampling

class RevertNew_P3(nn.Module):
    def __init__(self,input_channel=3, config=GlobalConfig()):
        super(RevertNew_P3, self).__init__()
        self.config = config
        self.alpha = 1.0
        # input channel: 3, output channel: 96

        """Features with Kernel Size 3---->channel:50 """
        self.pre_3 = SingleConv(2*input_channel, out_channels=32, kernel_size=3, stride=1, dilation=2, padding=2)
        self.Down1_pool_3 = SingleConv(32, out_channels=64, kernel_size=3, stride=2, dilation=1, padding=1)
        self.Down1_conv_3 = SingleConv(64, out_channels=64, kernel_size=3, stride=1, dilation=2, padding=2)
        self.Down2_pool_3 = SingleConv(64, out_channels=128, kernel_size=3, stride=2, dilation=1, padding=1)
        self.Down2_conv_3 = SingleConv(128, out_channels=128, kernel_size=3, stride=1, dilation=2, padding=2)
        self.Down3_pool_3 = SingleConv(128, out_channels=256, kernel_size=3, stride=2, dilation=1, padding=1)
        self.Down3_conv_3 = SingleConv(256, out_channels=256, kernel_size=3, stride=1, dilation=2, padding=2)
        self.Down2_dilate_conv0_3 = SingleConv(256, out_channels=256, kernel_size=3, stride=1, dilation=2, padding=2)
        self.Down2_dilate_conv1_3 = SingleConv(256, out_channels=256, kernel_size=3, stride=1, dilation=4, padding=4)
        self.Down2_dilate_conv2_3 = SingleConv(256, out_channels=256, kernel_size=3, stride=1, dilation=8, padding=8)
        self.Down2_dilate_conv3_3 = SingleConv(256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1)
        self.Down2_dilate_conv4_3 = SingleConv(256, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1)
        self.upsample2_3 = PureUpsampling(scale=2)
        self.Up2_conv_3 = SingleConv(384, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1)
        self.upsample1_3 = PureUpsampling(scale=2)
        self.Up1_conv_3 = SingleConv(192, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1)
        self.upsample0_3 = PureUpsampling(scale=2)
        self.Up0_conv_3 = SingleConv(96, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1)
        """Prep Network"""
        self.retrieve3_1 = SingleConv(32, 16, kernel_size=3, stride=1, dilation=1, padding=1)
        self.retrieve3_2 = SingleConv(16, 8, kernel_size=3, stride=1, dilation=1, padding=1)
        self.retrieve3_3 = nn.Sequential(
            nn.Conv2d(8, 3, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.Tanh()
        )


    def forward(self, p, p5_final):
        # Features with Kernel Size 3
        p3_in = torch.cat((p, p5_final), 1)
        p3_0 = self.pre_5(p3_in)
        p3_1 = self.Down1_pool_3(p3_0)
        p3_2 = self.Down1_conv_3(p3_1)
        p3_3 = self.Down2_pool_3(p3_2)
        p3_4 = self.Down2_conv_3(p3_3)
        p3_5 = self.Down3_pool_3(p3_4)
        p3_6 = self.Down3_conv_3(p3_5)
        p3_7 = self.Down2_dilate_conv0_3(p3_6)
        p3_8 = self.Down2_dilate_conv1_3(p3_7)
        p3_9 = self.Down2_dilate_conv2_3(p3_8)
        p3_10 = self.Down2_dilate_conv3_3(p3_9)
        p3_11 = self.Down2_dilate_conv4_3(p3_10)
        p3_12 = self.upsample2_3(p3_11)
        p3_12_cat = torch.cat((p3_12, p3_4), 1)
        p3_13 = self.Up2_conv_3(p3_12_cat)
        p3_14 = self.upsample1_3(p3_13)
        p3_14_cat = torch.cat((p3_14, p3_2), 1)
        p3_15 = self.Up1_conv_3(p3_14_cat)
        p3_16 = self.upsample0_3(p3_15)
        p3_16_cat = torch.cat((p3_16, p3_0), 1)
        p3_17 = self.Up0_conv_3(p3_16_cat)
        # concat = torch.cat((p7_12, p5_14, p3_16), 1)
        r1 = self.retrieve3_1(p3_17)
        r2 = self.retrieve3_2(r1)
        r3 = self.retrieve3_3(r2)
        return p7_final, p5_final, r3
