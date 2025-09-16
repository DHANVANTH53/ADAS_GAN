# models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64,128,256,512]):
        super().__init__()
        self.inc = DoubleConv(in_channels, features[0])
        self.downs = nn.ModuleList()
        for i in range(len(features)-1):
            self.downs.append(Down(features[i], features[i+1]))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.ups = nn.ModuleList()
        rev = list(reversed(features))
        in_up_channels = features[-1]*2
        for feat in rev:
            self.ups.append(Up(in_up_channels, feat))
            in_up_channels = feat
        self.outc = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()  # assuming input normalized to [0,1]
        )

    def forward(self, x):
        x1 = self.inc(x)
        features = [x1]
        for d in self.downs:
            features.append(d(features[-1]))
        x_bot = self.bottleneck(features[-1])
        x = x_bot
        for i, up in enumerate(self.ups):
            x = up(x, features[-2-i])
        return self.outc(x)
