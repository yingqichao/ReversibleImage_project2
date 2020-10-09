import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
import util
from network.single_conv import SingleConv
from network.pure_upsample import PureUpsampling

class Revert(nn.Module):
    def __init__(self,config=GlobalConfig()):
        super(Revert, self).__init__()
        self.config = config
        self.downsample_8 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ELU(inplace=True)
        )
        # 128
        self.downsample_7 = SingleConv(64, out_channels=128, kernel_size=5, stride=2, dilation=1, padding=2)
        # self.Down1_conv_7 = SingleConv(64, out_channels=64, kernel_size=7, stride=1, dilation=1, padding=3)
        # 64
        self.downsample_6 = SingleConv(128, out_channels=256, kernel_size=5, stride=2, dilation=1, padding=2)
        # self.Down2_conv_7 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=1, padding=3)
        # 32
        self.downsample_5 = SingleConv(256, out_channels=512, kernel_size=5, stride=2, dilation=1, padding=2)
        # 16
        self.downsample_4 = SingleConv(512, out_channels=512, kernel_size=5, stride=2, dilation=1, padding=2)
        # 8
        self.downsample_3 = SingleConv(512, out_channels=512, kernel_size=5, stride=2, dilation=1, padding=2)
        # 4
        self.downsample_2 = SingleConv(512, out_channels=512, kernel_size=5, stride=2, dilation=1, padding=2)
        # 2
        self.downsample_1 = SingleConv(512, out_channels=512, kernel_size=5, stride=2, dilation=1, padding=2)
        # 1
        self.downsample_0 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=5, stride=2, dilation=1, padding=2),
            nn.ELU(inplace=True)
        )
        # 2
        self.upsample8_3 = nn.Sequential(
            PureUpsampling(scale=2),
            SingleConv(512, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 4
        self.upsample7_3 = nn.Sequential(
            PureUpsampling(scale=2),
            SingleConv(1024, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 8
        self.upsample6_3 = nn.Sequential(
            PureUpsampling(scale=2),
            SingleConv(1024, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 16
        self.upsample5_3 = nn.Sequential(
            PureUpsampling(scale=2),
            SingleConv(1024, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 32
        self.upsample4_3 = nn.Sequential(
            PureUpsampling(scale=2),
            SingleConv(1024, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 64
        self.upsample3_3 = nn.Sequential(
            PureUpsampling(scale=2),
            SingleConv(1024, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 128
        self.upsample2_3 = nn.Sequential(
            PureUpsampling(scale=2),
            SingleConv(512, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 256
        self.upsample1_3 = nn.Sequential(
            PureUpsampling(scale=2),
            SingleConv(256, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1)
        )

        self.finalH = nn.Sequential(
            # SingleConv(128, out_channels=128, kernel_size=5, stride=1, dilation=1, padding=2),
            nn.Conv2d(128, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )
        self.finalH2 = nn.Sequential(
            # SingleConv(6, out_channels=6, kernel_size=5, stride=1, dilation=1, padding=2),
            nn.Conv2d(6, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )

        self.final32 = nn.Sequential(
            nn.Conv2d(512, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )
        self.final64 = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )
        self.final128 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=1, padding=0),
            nn.Tanh()
        )
        # self.final256 = nn.Sequential(
        #     nn.Conv2d(64, 3, kernel_size=1, padding=0),
        #     nn.Tanh()
        # )
        # self.final256 = nn.Sequential(
        #     nn.Conv2d(64, 3, kernel_size=1, padding=0),
        #     nn.Tanh()
        # )
        self.upsample2 = PureUpsampling(scale=2)


    def forward(self, ori_image, stage):
        # 阶梯训练，仿照ProgressiveGAN
        down8 = self.downsample_8(ori_image)
        down7 = self.downsample_7(down8)
        down6 = self.downsample_6(down7)
        down5 = self.downsample_5(down6)
        down4 = self.downsample_4(down5)
        down3 = self.downsample_3(down4)
        down2 = self.downsample_2(down3)
        down1 = self.downsample_1(down2)
        down0 = self.downsample_0(down1)
        up8 = self.upsample8_3(down0)
        up8_cat = torch.cat((down1, up8), 1)
        up7 = self.upsample7_3(up8_cat)
        up7_cat = torch.cat((down2, up7), 1)
        up6 = self.upsample6_3(up7_cat)
        up6_cat = torch.cat((down3, up6), 1)
        up5 = self.upsample5_3(up6_cat)
        up5_cat = torch.cat((down4, up5), 1)

        if stage >= 32:
            up4 = self.upsample4_3(up5_cat)
            out_32 = self.final32(up4)
            if stage==32:
                return out_32
        if stage >= 64:
            up_64 = self.upsample2(out_32)
            up4_cat = torch.cat((down5, up4), 1)
            up3 = self.upsample3_3(up4_cat)
            out_64 = self.final64(up3)
            if stage==64:
                return up_64, out_64
        if stage >= 128:
            up_128 = self.upsample2(out_64)
            up3_cat = torch.cat((down6, up3), 1)
            up2 = self.upsample2_3(up3_cat)
            out_128 = self.final128(up2)
            if stage == 128:
                return up_128, out_128
        if stage >= 256:
            up_256 = self.upsample2(out_128)
            up2_cat = torch.cat((down7, up2), 1)
            up1 = self.upsample1_3(up2_cat)
            up1_cat = torch.cat((down8, up1), 1)
            out_256 = self.finalH(up1_cat)
            out_cat = torch.cat((out_256, ori_image), 1)
            result = self.finalH2(out_cat)
            if stage == 256:
                return up_256, result
        if stage == 512:

            out_cat = torch.cat((out_256, ori_image), 1)
            result = self.finalH2(out_cat)
            return out_256, result

        # Won't reach
        return None
