 # models/restormer.py
"""
A transformer U-Net style restoration network.
Encoder/decoder are convolutional (like U-Net).
At the bottleneck we apply patch embedding and a stack of Transformer encoder layers
(so global context is captured on a reduced token resolution).
This keeps memory moderate but is powerful for restoration tasks compared to plain U-Net.

Designed to be "heavy" (good performance) — tune dims for your GPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if necessary
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Patch embedding: conv -> flatten patches
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=512, embed_dim=512, patch_size=16):
        super().__init__()
        # patch_size is spatial reduction factor
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: B,C,H,W -> z: B,embed_dim, H/patch, W/patch
        z = self.proj(x)               # B,embed,Hp,Wp
        B, C, Hp, Wp = z.shape
        z = z.flatten(2).transpose(1,2) # B, N=Hp*Wp, C
        z = self.norm(z)
        return z, (Hp, Wp)

class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=512, out_ch=512, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_ch, kernel_size=patch_size, stride=patch_size)

    def forward(self, z, hw):
        # z: B, N, C  -> B,C,Hp,Wp -> convtranspose -> B,out_ch,H,W
        B, N, C = z.shape
        Hp, Wp = hw
        z = z.transpose(1,2).view(B, C, Hp, Wp)
        x = self.proj(z)
        return x

# Transformer encoder wrapper (uses nn.TransformerEncoderLayer)
class TransformerStack(nn.Module):
    def __init__(self, embed_dim=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        # x: B,N,C
        return self.encoder(x)

class RestormerUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64,128,256,512], transformer_dim=512, patch_size=8, nhead=8, num_transformer_layers=8, ff_dim=2048):
        """
        - features: encoder-decoder channel sizes (like U-Net)
        - transformer_dim: token embedding dim at bottleneck
        - patch_size: how much spatial reduction before transformer (power of 2)
        - nhead / num_transformer_layers / ff_dim: transformer params
        """
        super().__init__()
        # encoder
        self.inc = ConvBlock(in_channels, features[0])
        self.downs = nn.ModuleList()
        for i in range(len(features)-1):
            self.downs.append(Down(features[i], features[i+1]))

        # bottleneck conv to produce channels matching transformer input
        bottleneck_in_ch = features[-1]
        self.bottleneck_conv = ConvBlock(bottleneck_in_ch, bottleneck_in_ch)
        # patch embed expects in_ch = bottleneck channels
        self.patch_embed = PatchEmbed(in_ch=bottleneck_in_ch, embed_dim=transformer_dim, patch_size=patch_size)
        self.transformer = TransformerStack(embed_dim=transformer_dim, nhead=nhead, num_layers=num_transformer_layers, dim_feedforward=ff_dim)
        self.patch_unembed = PatchUnEmbed(embed_dim=transformer_dim, out_ch=bottleneck_in_ch, patch_size=patch_size)

        # decoder
        rev = list(reversed(features))
        self.ups = nn.ModuleList()
        in_up = bottleneck_in_ch
        for feat in rev:
            self.ups.append(Up(in_up, feat))
            in_up = feat

        self.outc = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x)             # features[0]
        feat_list = [x1]
        xi = x1
        for d in self.downs:
            xi = d(xi)
            feat_list.append(xi)
        # bottleneck conv
        x_bot = self.bottleneck_conv(feat_list[-1])  # B, Cb, H, W

        # patch embed -> transformer
        tokens, (Hp, Wp) = self.patch_embed(x_bot)   # B, N, C_t
        tokens = self.transformer(tokens)            # global context -> B,N,C_t

        # unembed tokens -> back to spatial
        x_trans = self.patch_unembed(tokens, (Hp, Wp))  # B, Cb, H, W

        # decoder using features from encoder (skip connections)
        x = x_trans
        for i, up in enumerate(self.ups):
            # features list: [x1, x2, ..., xL], we need to index from end
            skip = feat_list[-2 - i]
            x = up(x, skip)

        out = self.outc(x)
        return out

