import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
import util

class PrepNetwork_Unet(nn.Module):
    def __init__(self,input_channel, config=GlobalConfig()):
        super(PrepNetwork_Unet, self).__init__()
        self.config = config
        # Size: 256->128
        self.Down1_conv = DoubleConv(input_channel, 64)
        self.Down1_pool = nn.MaxPool2d(2)

        # Size: 128->64
        self.Down2_conv = DoubleConv(64, 128)
        self.Down2_pool = nn.MaxPool2d(2)

        # Size: 64->32
        self.Down3_conv = DoubleConv(128, 256)
        self.Down3_pool = nn.MaxPool2d(2)

        # Size: 32->16
        self.Down4_conv = DoubleConv(256, 512)
        self.Down4_pool = nn.MaxPool2d(2)

        # self.Conv5 = nn.Sequential(
        #     DoubleConv(512, 1024),
        #     DoubleConv(1024, 1024),
        # )

        self.hiding_1_1 = nn.Sequential(
            DoubleConv(512, 512, mode=0),
            DoubleConv(512, 512, mode=0),
        )
        self.hiding_1_2 = nn.Sequential(
            DoubleConv(512, 512, mode=1),
            DoubleConv(512, 512, mode=1),
        )

        # Size:16->32
        self.Up4_convT = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.Up4_conv = DoubleConv(1024, 512)

        # Size:32->64
        self.Up3_convT = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.Up3_conv = DoubleConv(512, 256)
        # Size:64->128
        self.Up2_convT = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.Up2_conv = DoubleConv(256, 128)
        # Size:128->256
        self.Up1_convT = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.Up1_conv = DoubleConv(128, 64)
        # Prep Network
        self.prep1_1 = nn.Sequential(
            DoubleConv(64, 50,mode=0),
            DoubleConv(50, 50,mode=0))
        self.prep1_2 = nn.Sequential(
            DoubleConv(64, 50, mode=1),
            DoubleConv(50, 50, mode=1))
        self.prep1_3 = nn.Sequential(
            DoubleConv(64, 50, mode=2),
            DoubleConv(50, 50, mode=2))
        self.prep2_1 = nn.Sequential(
            DoubleConv(150, 50, mode=0),
            DoubleConv(50, 50, mode=0))
        self.prep2_2 = nn.Sequential(
            DoubleConv(150, 50, mode=1),
            DoubleConv(50, 50, mode=1))
        self.prep2_3 = nn.Sequential(
            DoubleConv(150, 50, mode=2),
            DoubleConv(50, 50, mode=2))


    def forward(self, p):
        # Size: 256->128
        down1_c = self.Down1_conv(p)
        down1_p = self.Down1_pool(down1_c)

        # Size: 128->64
        down2_c = self.Down2_conv(down1_p)
        down2_p = self.Down2_pool(down2_c)

        # Size: 64->32
        down3_c = self.Down3_conv(down2_p)
        down3_p = self.Down3_pool(down3_c)

        # Size: 32->16
        down4_c = self.Down4_conv(down3_p)
        down4_p = self.Down4_pool(down4_c)

        hid_1 = self.hiding_1_1(down4_p)
        hid_2 = self.hiding_1_2(down4_p)
        mid = torch.cat([hid_1, hid_2], dim=1)

        up4_convT = self.Up4_convT(mid)
        merge4 = torch.cat([up4_convT, down4_c], dim=1)
        up4_conv = self.Up4_conv(merge4)
        # Size: 32->64
        up3_convT = self.Up3_convT(up4_conv)
        merge3 = torch.cat([up3_convT, down3_c], dim=1)
        up3_conv = self.Up3_conv(merge3)
        # Size: 64->128
        up2_convT = self.Up2_convT(up3_conv)
        merge2 = torch.cat([up2_convT, down2_c], dim=1)
        up2_conv = self.Up2_conv(merge2)
        # Size: 128->256
        up1_convT = self.Up1_convT(up2_conv)
        merge1 = torch.cat([up1_convT, down1_c], dim=1)
        up1_conv = self.Up1_conv(merge1)
        # Prepare
        p1 = self.prep1_1(up1_conv)
        p2 = self.prep1_2(up1_conv)
        p3 = self.prep1_3(up1_conv)
        mid = torch.cat((p1, p2, p3), 1)
        p4 = self.prep2_1(mid)
        p5 = self.prep2_2(mid)
        p6 = self.prep2_3(mid)
        out = torch.cat((p4, p5, p6), 1)
        return out