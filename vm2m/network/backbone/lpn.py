import torch
import torch.nn as nn
import torch.nn.functional as F

from .mobilenetv2 import MobileNetV2Backbone

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar,mode='bilinear'):
    src = F.interpolate(src,size=tar.shape[2:],mode=mode,align_corners=False if mode=='bilinear' else None)
    return src
upas = _upsample_like

class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        n, c, h, w = bn_x.shape
        if n==1 and h==1 and w==1:
            in_x = self.inorm(x[:, self.inorm_channels:, ...].contiguous().expand(n*2, c, h, w).contiguous())[0:1]
        else:
            in_x = self.inorm(x[:, self.inorm_channels:, ...].contiguous())
        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias)
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w.expand_as(x)


class HLBranch(nn.Module):
    """ High Resolution Branch of MODNet
    """

    def __init__(self, in_channels, hr_channels, enc_channels, with_norm=True):
        super(HLBranch, self).__init__()

        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)

        self.p32x = Conv2dIBNormRelu(enc_channels[4], 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False)

        self.conv_dec16x = nn.Sequential(
            Conv2dIBNormRelu(enc_channels[4]+enc_channels[3], 2*hr_channels, 3, stride=1, padding=1, with_ibn=with_norm),
            Conv2dIBNormRelu(2*hr_channels, hr_channels, 3, stride=1, padding=1, with_ibn=with_norm),
        )
        self.p16x = Conv2dIBNormRelu(hr_channels+1, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False)

        self.conv_dec8x = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + enc_channels[2], 2*hr_channels, 3, stride=1, padding=1, with_ibn=with_norm),
            Conv2dIBNormRelu(2*hr_channels, hr_channels, 3, stride=1, padding=1, with_ibn=with_norm),
        )
        self.p8x = Conv2dIBNormRelu(hr_channels+1, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False)

        self.conv_dec4x = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + enc_channels[1], 2*hr_channels, 3, stride=1, padding=1, with_ibn=with_norm),
            Conv2dIBNormRelu(2*hr_channels, hr_channels, 3, stride=1, padding=1, with_ibn=with_norm),
        )
        self.p4x = Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False)

        self.conv_dec2x = nn.Sequential(
            Conv2dIBNormRelu(hr_channels+enc_channels[0], 2*hr_channels, 3, stride=1, padding=1, with_ibn=with_norm),
            Conv2dIBNormRelu(2*hr_channels, hr_channels, 3, stride=1, padding=1, with_ibn=with_norm),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1, with_ibn=with_norm),
        )
        self.p2x = Conv2dIBNormRelu(hr_channels+1, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False)

        self.conv_dec1x = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + in_channels, hr_channels, 3, stride=1, padding=1, with_ibn=with_norm),
        )
        self.p1x = Conv2dIBNormRelu(hr_channels+1, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False)

        self.p0x = Conv2dIBNormRelu(2, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False)

    def forward(self, img, enc2x, enc4x, enc8x, enc16x, enc32x, is_training=True):
        enc32x = self.se_block(enc32x)
        p32x = self.p32x(enc32x)
        p32x = upas(p32x, img)

        dec16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False)
        dec16x = self.conv_dec16x(torch.cat((dec16x, enc16x), dim=1))
        p16x = self.p16x(torch.cat((dec16x, upas(p32x, dec16x)), dim=1))
        p16x = upas(p16x, img)

        dec8x = F.interpolate(dec16x, scale_factor=2, mode='bilinear', align_corners=False)
        dec8x = self.conv_dec8x(torch.cat((dec8x, enc8x), dim=1))
        p8x = self.p8x(torch.cat((dec8x, upas(p16x, dec8x)), dim=1))
        p8x = upas(p8x, img)

        dec4x = F.interpolate(dec8x, scale_factor=2, mode='bilinear', align_corners=False)
        dec4x = self.conv_dec4x(torch.cat((dec4x, enc4x), dim=1))
        p4x = self.p4x(dec4x)
        p4x = upas(p4x, img)

        dec2x = F.interpolate(dec4x, scale_factor=2, mode='bilinear', align_corners=False)
        dec2x = self.conv_dec2x(torch.cat((dec2x, enc2x), dim=1))
        p2x = self.p2x(torch.cat((dec2x, upas(p4x, dec2x)), dim=1))
        p2x = upas(p2x, img)

        dec1x = F.interpolate(dec2x, scale_factor=2, mode='bilinear', align_corners=False)
        dec1x = self.conv_dec1x(torch.cat((dec1x, img), dim=1))
        p1x = self.p1x(torch.cat((dec1x, upas(p2x, dec1x)), dim=1))

        p0x = self.p0x(torch.cat((p1x, upas(p8x, p1x)), dim=1))

        seg_out = [torch.sigmoid(p) for p in (p8x, p16x, p32x)]
        mat_out = [torch.sigmoid(p) for p in (p1x, p2x, p4x)]
        fus_out = [torch.sigmoid(p) for p in (p0x,)]
        return seg_out, mat_out, fus_out, [dec1x, dec2x, dec4x, dec8x, dec16x]


class AuxilaryHead(nn.Module):
    def __init__(self, hr_channels, enc_channels):
        super().__init__()

        self.s1 = Conv2dIBNormRelu(
                hr_channels, 3, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False)
        self.s2 = Conv2dIBNormRelu(
                hr_channels, 3, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False)
        self.s4 = Conv2dIBNormRelu(
                hr_channels, 3, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False)
        self.s8 = Conv2dIBNormRelu(
                hr_channels, 3, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False)
        self.s16 = Conv2dIBNormRelu(
                hr_channels, 3, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False)

    def forward(self, dec1x, dec2x, dec4x, dec8x, dec16x, is_training=True):
        p1 = self.s1(dec1x)

        x2 = self.s2(dec2x)
        p2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)

        x4 = self.s4(dec4x)
        p4 = F.interpolate(x4, scale_factor=4, mode='bilinear', align_corners=False)

        x8 = self.s8(dec8x)
        p8 = F.interpolate(x8, scale_factor=8, mode='bilinear', align_corners=False)

        x16 = self.s16(dec16x)
        p16 = F.interpolate(x16, scale_factor=16, mode='bilinear', align_corners=False)

        return (p1,p2,p4,p8,p16)


class LPN(nn.Module):
    def __init__(self, in_chn=3, mid_chn=128):
        super().__init__()
        self.backbone = MobileNetV2Backbone(in_chn)
        self.decoder = HLBranch(in_chn, mid_chn, self.backbone.enc_channels)
        # self.aux_head = AuxilaryHead(mid_chn, self.backbone.enc_channels)

    def forward(self, images):
        enc2x, enc4x, enc8x, enc16x, enc32x = self.backbone(images)
        seg_outs, mat_outs, fus_outs, decoded_feats = self.decoder(images, enc2x, enc4x, enc8x, enc16x, enc32x)
        return fus_outs[0], decoded_feats[-1]
    
def lpn(num_mask=1, mid_chn=32, **kwargs):
    return LPN(in_chn=3 + num_mask, mid_chn=mid_chn)