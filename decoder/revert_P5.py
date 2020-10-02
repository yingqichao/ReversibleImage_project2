from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
import util
import torch
import torch.nn as nn
from network.single_conv import SingleConv
from network.pure_upsample import PureUpsampling

class RevertNew_P5(nn.Module):
    def __init__(self,input_channel=3, config=GlobalConfig()):
        super(RevertNew_P5, self).__init__()
        self.config = config
        self.alpha = 1.0
        # input channel: 3, output channel: 96

        """Features with Kernel Size 5---->channel:75 """
        self.pre_5 = SingleConv(2*input_channel, out_channels=32, kernel_size=5, stride=1, dilation=1, padding=2)
        self.Down1_pool_5 = SingleConv(32, out_channels=64, kernel_size=5, stride=2, dilation=1, padding=2)
        self.Down1_conv_5 = SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2)
        self.Down2_pool_5 = SingleConv(64, out_channels=128, kernel_size=5, stride=2, dilation=1, padding=2)
        self.Down2_conv_5 = SingleConv(128, out_channels=128, kernel_size=5, stride=1, dilation=1, padding=2)
        self.Down3_pool_5 = SingleConv(128, out_channels=256, kernel_size=5, stride=2, dilation=1, padding=2)
        self.Down3_conv_5 = SingleConv(256, out_channels=256, kernel_size=5, stride=1, dilation=1, padding=2)
        self.Down2_dilate_conv0_5 = SingleConv(256, out_channels=256, kernel_size=5, stride=1, dilation=2, padding=4)
        self.Down2_dilate_conv1_5 = SingleConv(256, out_channels=256, kernel_size=5, stride=1, dilation=4, padding=8)
        self.Down2_dilate_conv2_5 = SingleConv(256, out_channels=256, kernel_size=5, stride=1, dilation=8, padding=16)
        self.Down2_dilate_conv3_5 = SingleConv(256, out_channels=256, kernel_size=5, stride=1, dilation=1, padding=2)
        self.Down2_dilate_conv4_5 = SingleConv(256, out_channels=256, kernel_size=5, stride=1, dilation=1, padding=2)
        self.upsample2_5 = PureUpsampling(scale=4)
        self.Up2_conv_5 = SingleConv(320, out_channels=128, kernel_size=5, stride=1, dilation=1, padding=2)
        self.upsample1_5 = PureUpsampling(scale=2)
        self.Up1_conv_5 = SingleConv(160, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2)
        self.retrieve5_1 = SingleConv(64, 32, kernel_size=3, stride=1, dilation=1, padding=1)
        self.retrieve5_2 = SingleConv(32, 16, kernel_size=3, stride=1, dilation=1, padding=1)
        self.final5 = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1, padding=0))



    def forward(self, p, p7_final):
        # Features with Kernel Size 7

        # Features with Kernel Size 5
        p5_in = torch.cat((p, p7_final), 1)
        p5_0 = self.pre_5(p5_in)
        p5_1 = self.Down1_pool_5(p5_0)
        p5_2 = self.Down1_conv_5(p5_1)
        p5_3 = self.Down2_pool_5(p5_2)
        p5_4 = self.Down2_conv_5(p5_3)
        p5_5 = self.Down3_pool_5(p5_4)
        p5_6 = self.Down3_conv_5(p5_5)
        p5_7 = self.Down2_dilate_conv0_5(p5_6)
        p5_8 = self.Down2_dilate_conv1_5(p5_7)
        p5_9 = self.Down2_dilate_conv2_5(p5_8)
        p5_10 = self.Down2_dilate_conv3_5(p5_9)
        p5_11 = self.Down2_dilate_conv4_5(p5_10)
        p5_12 = self.upsample2_5(p5_11)
        p5_12_cat = torch.cat((p5_12, p5_2), 1)
        p5_13 = self.Up2_conv_5(p5_12_cat)
        p5_14 = self.upsample1_5(p5_13)
        p5_14_cat = torch.cat((p5_14, p5_0), 1)
        p5_15 = self.Up1_conv_5(p5_14_cat)
        r51 = self.retrieve5_1(p5_15)
        r52 = self.retrieve5_2(r51)
        p5_final = self.final5(r52)

        #p5_out = p7_final*self.alpha+p5_final*(1-self.alpha)

        return p5_final

