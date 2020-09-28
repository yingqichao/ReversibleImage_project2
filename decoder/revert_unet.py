import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
import util
from network.single_conv import SingleConv
from network.pure_upsample import PureUpsampling

class Revert_Unet(nn.Module):
    def __init__(self,input_channel=96, config=GlobalConfig()):
        super(Revert_Unet, self).__init__()
        self.config = config
        # input channel: 3, output channel: 96
        # Features with Kernel Size 7
        self.pre_7 = SingleConv(input_channel, out_channels=32, kernel_size=7, stride=1, dilation=2, padding=6)
        self.Down1_pool_7 = SingleConv(32, out_channels=64, kernel_size=7, stride=2, dilation=1, padding=3)
        self.Down1_conv_7 = SingleConv(64, out_channels=64, kernel_size=7, stride=1, dilation=2, padding=6)
        self.Down2_pool_7 = SingleConv(64, out_channels=128, kernel_size=7, stride=2, dilation=1, padding=3)
        self.Down2_conv_7 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=2, padding=6)
        # self.Down2_dilate_conv1 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=2)
        # self.Down2_dilate_conv2 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=4)
        # self.Down2_dilate_conv3 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=8)
        # self.Down2_dilate_conv4 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=16)
        self.Down2_dilate_conv1_7 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=1, padding=3)
        self.Down2_dilate_conv2_7 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=1, padding=3)
        self.upsample2_7 = PureUpsampling(scale=2)
        self.Up2_conv_7 = SingleConv(128, out_channels=64, kernel_size=7, stride=1, dilation=1, padding=3)
        self.upsample1_7 = PureUpsampling(scale=2)
        self.Up1_conv_7 = SingleConv(64, out_channels=32, kernel_size=7, stride=1, dilation=1, padding=3)

        # Features with Kernel Size 5
        self.pre_5 = SingleConv(input_channel, out_channels=32, kernel_size=5, stride=1, dilation=2, padding=4)
        self.Down1_pool_5 = SingleConv(32, out_channels=64, kernel_size=5, stride=2, dilation=1, padding=2)
        self.Down1_conv_5 = SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=2, padding=4)
        self.Down2_pool_5 = SingleConv(64, out_channels=128, kernel_size=5, stride=2, dilation=1, padding=2)
        self.Down2_conv_5 = SingleConv(128, out_channels=128, kernel_size=5, stride=1, dilation=2, padding=4)
        # self.Down2_dilate_conv1 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=2)
        # self.Down2_dilate_conv2 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=4)
        # self.Down2_dilate_conv3 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=8)
        # self.Down2_dilate_conv4 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=16)
        self.Down2_dilate_conv1_5 = SingleConv(128, out_channels=128, kernel_size=5, stride=1, dilation=1, padding=2)
        self.Down2_dilate_conv2_5 = SingleConv(128, out_channels=128, kernel_size=5, stride=1, dilation=1, padding=2)
        self.upsample2_5 = PureUpsampling(scale=2)
        self.Up2_conv_5 = SingleConv(128, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2)
        self.upsample1_5 = PureUpsampling(scale=2)
        self.Up1_conv_5 = SingleConv(64, out_channels=32, kernel_size=5, stride=1, dilation=1, padding=2)
        # Features with Kernel Size 3
        self.pre_3 = SingleConv(input_channel, out_channels=32, kernel_size=3, stride=1, dilation=2, padding=2)
        self.Down1_pool_3 = SingleConv(32, out_channels=64, kernel_size=3, stride=2, dilation=1, padding=1)
        self.Down1_conv_3 = SingleConv(64, out_channels=64, kernel_size=3, stride=1, dilation=2, padding=2)
        self.Down2_pool_3 = SingleConv(64, out_channels=128, kernel_size=3, stride=2, dilation=1, padding=1)
        self.Down2_conv_3 = SingleConv(128, out_channels=128, kernel_size=3, stride=1, dilation=2, padding=2)
        # self.Down2_dilate_conv1 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=2)
        # self.Down2_dilate_conv2 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=4)
        # self.Down2_dilate_conv3 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=8)
        # self.Down2_dilate_conv4 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=16)
        self.Down2_dilate_conv1_3 = SingleConv(128, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1)
        self.Down2_dilate_conv2_3 = SingleConv(128, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1)
        self.upsample2_3 = PureUpsampling(scale=2)
        self.Up2_conv_3 = SingleConv(128, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1)
        self.upsample1_3 = PureUpsampling(scale=2)
        self.Up1_conv_3 = SingleConv(64, out_channels=32, kernel_size=3, stride=1, dilation=1, padding=1)

        self.finalH = nn.Sequential(
            nn.Conv2d(96, 3, kernel_size=1, padding=0))


    def forward(self, p, ori_image):
        # Features with Kernel Size 7
        p7_1 = self.pre_7(p)
        p7_2 = self.Down1_pool_7(p7_1)
        p7_3 = self.Down1_conv_7(p7_2)
        p7_4 = self.Down2_pool_7(p7_3)
        p7_5 = self.Down2_conv_7(p7_4)
        p7_6 = self.Down2_dilate_conv1_7(p7_5)
        p7_7 = self.Down2_dilate_conv2_7(p7_6)
        p7_8 = self.upsample2_7(p7_7)
        p7_9 = self.Up2_conv_7(p7_8)
        p7_10 = self.upsample1_7(p7_9)
        p7_11 = self.Up1_conv_7(p7_10)
        # Features with Kernel Size 5
        p5_1 = self.pre_5(p)
        p5_2 = self.Down1_pool_5(p5_1)
        p5_3 = self.Down1_conv_5(p5_2)
        p5_4 = self.Down2_pool_5(p5_3)
        p5_5 = self.Down2_conv_5(p5_4)
        p5_6 = self.Down2_dilate_conv1_5(p5_5)
        p5_7 = self.Down2_dilate_conv2_5(p5_6)
        p5_8 = self.upsample2_5(p5_7)
        p5_9 = self.Up2_conv_5(p5_8)
        p5_10 = self.upsample1_5(p5_9)
        p5_11 = self.Up1_conv_5(p5_10)
        # Features with Kernel Size 3
        p3_1 = self.pre_3(p)
        p3_2 = self.Down1_pool_3(p3_1)
        p3_3 = self.Down1_conv_3(p3_2)
        p3_4 = self.Down2_pool_3(p3_3)
        p3_5 = self.Down2_conv_3(p3_4)
        p3_6 = self.Down2_dilate_conv1_3(p3_5)
        p3_7 = self.Down2_dilate_conv2_3(p3_6)
        p3_8 = self.upsample2_3(p3_7)
        p3_9 = self.Up2_conv_3(p3_8)
        p3_10 = self.upsample1_3(p3_9)
        p3_11 = self.Up1_conv_3(p3_10)

        out = torch.cat((p7_11, p5_11, p3_11), 1)
        recover = self.finalH(out)
        return recover
