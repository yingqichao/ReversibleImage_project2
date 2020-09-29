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
        self.prep = DoubleConv(3, 16)
        self.upsample2 = PureUpsampling(scale=2)
        # 512-> 256
        self.Conv512_256 = nn.Sequential(
            DoubleConv(16, 32),
            DoubleConv(32, 32),
            PureUpsampling(scale=0.5),
        )
        # 256 -> 128
        self.Conv256_128 = nn.Sequential(
            DoubleConv(32, 64),
            DoubleConv(64, 64),
            PureUpsampling(scale=0.5),
        )
        # 128 -> 64
        self.Conv128_64 = nn.Sequential(
            DoubleConv(64, 128),
            DoubleConv(128, 128),
            PureUpsampling(scale=0.5),
        )
        # 64-> 32
        self.Conv64_32 = nn.Sequential(
            DoubleConv(128, 256),
            DoubleConv(256, 256),
            PureUpsampling(scale=0.5),
        )
        # mid
        self.mid = nn.Sequential(
            DoubleConv(256, 256),
            DoubleConv(256, 256),
        )
        # 32->64
        self.Conv32_64 = nn.Sequential(
            PureUpsampling(scale=2),
            DoubleConv(256, 128),
            DoubleConv(128, 128),
        )
        # 64->128
        self.Conv64_128 = nn.Sequential(
            PureUpsampling(scale=2),
            DoubleConv(128, 64),
            DoubleConv(64, 64),
        )
        # 128->256
        self.Conv128_256 = nn.Sequential(
            PureUpsampling(scale=2),
            DoubleConv(64, 32),
            DoubleConv(32, 32),
        )
        # 256->512
        self.Conv256_512 = nn.Sequential(
            PureUpsampling(scale=2),
            DoubleConv(32, 16),
            DoubleConv(16, 16),
        )

        self.final32 = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1, padding=0))
        self.final64 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=1, padding=0))
        self.final128 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, padding=0))
        self.final256 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, padding=0))
        self.final512 = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1, padding=0))


    def forward(self, ori_image, stage):
        # 阶梯训练，仿照ProgressiveGAN
        p = self.prep(ori_image)
        p_256 = self.Conv512_256(p)
        p_128 = self.Conv256_128(p_256)
        p_64 = self.Conv128_64(p_128)
        p_32 = self.Conv64_32(p_64)
        if stage >= 32:
            res_32 = self.mid(p_32)
            out_32 = self.final32(res_32)
            if stage==32:
                return out_32
        if stage>=64:
            res_64 = self.Conv32_64(res_32)
            up_64 = self.upsample2(out_32)
            in_64 = torch.cat((res_64,up_64), 1)
            out_64 = self.final64(in_64)
            if stage==64:
                return out_64
        if stage >= 128:
            res_128 = self.Conv64_128(res_64)
            up_128 = self.upsample2(out_64)
            in_128 = torch.cat((res_128, up_128), 1)
            out_128 = self.final64(in_128)
            if stage == 128:
                return out_128
        if stage >= 256:
            res_256 = self.Conv128_256(res_128)
            up_256 = self.upsample2(out_128)
            in_256 = torch.cat((res_256, up_256), 1)
            out_256 = self.final64(in_256)
            if stage == 256:
                return out_256
        if stage >= 512:
            res_512 = self.Conv64_128(res_256)
            up_512 = self.upsample2(out_256)
            in_512 = torch.cat((res_512, up_512), 1)
            out_512 = self.final64(in_512)
            if stage == 512:
                return out_512
        # Won't reach
        return None
