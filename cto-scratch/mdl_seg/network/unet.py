import torch
import torch.nn as nn

from .blocks import ConvBNR, resize_to


def double_conv(in_ch, out_ch):
    return nn.Sequential(ConvBNR(in_ch, out_ch), ConvBNR(out_ch, out_ch))


class UNet(nn.Module):
    def __init__(self, num_class=2, in_ch=3, base=32):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8, base * 16]
        self.downs = nn.ModuleList()
        prev = in_ch
        for ch in chs:
            self.downs.append(double_conv(prev, ch))
            prev = ch
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(prev, ch, 2, stride=2))
            self.up_convs.append(double_conv(ch * 2, ch))
            prev = ch
        self.head = nn.Conv2d(prev, num_class, 1)

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = resize_to(up(x), skip)
            x = conv(torch.cat([x, skip], dim=1))
        return [self.head(x)], None
