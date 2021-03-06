import torch
import torch.nn as nn

from config import GlobalConfig
from network.double_conv import DoubleConv


class Decoder_pool(nn.Module):
    def __init__(self,config=GlobalConfig()):
        super(Decoder_pool, self).__init__()
        self.config = config
        # Size: 256->128
        self.Down1_conv = DoubleConv(3, 64)
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

        self.Conv5 = nn.Sequential(
            DoubleConv(512, 1024),
            DoubleConv(1024, 1024),
            # DoubleConv(1024, 1024),
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
        # 最后一个卷积层得到输出
        self.final_conv = nn.Conv2d(64+3, 3, 1)
        #self.final_conv = nn.Conv2d(6, 3, 1)


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

        mid = self.Conv5(down4_p)

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
        merge0 = torch.cat([up1_conv, p], dim=1)
        out = self.final_conv(merge0)
        return out