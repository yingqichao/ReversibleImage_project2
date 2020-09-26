import torch.nn as nn

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, mode=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if mode==0:
            kernel_size_1, padding_size_1 = 3, 1
        elif mode==1:
            kernel_size_1, padding_size_1 = 5, 2
        elif mode==2:
            kernel_size_1, padding_size_1 = 7, 3
        else:
            kernel_size_1, padding_size_1 = 9, 4

        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=kernel_size_1, padding=padding_size_1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)

