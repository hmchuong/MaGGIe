import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

from vm2m.network.backbone.sparse_resnet_bn import l_sparse_resnet18

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
    

class SparseCAM(nn.Module):
    def __init__(self, local_inc, global_inc, with_norm=True):
        super(SparseCAM, self).__init__()

        self.pool_fg = nn.AdaptiveAvgPool2d(1)
        self.pool_bg = nn.AdaptiveAvgPool2d(1)
        self.conv_f = Conv2dIBNormRelu(global_inc, global_inc, kernel_size=1, with_ibn=False)
        self.conv_b = Conv2dIBNormRelu(global_inc, global_inc, kernel_size=1, with_ibn=False)
        self.conv_g = Conv2dIBNormRelu(2*global_inc, local_inc, kernel_size=1, with_relu=False, with_ibn=False)

    def forward(self, idx, x, ctx, mask):
        mask_lr = F.interpolate(mask, ctx.size()[2:], align_corners=False, mode='bilinear')
        fg_pool = self.pool_fg(ctx * mask_lr)
        fg_ctx = self.conv_f(fg_pool)
        bg_pool = self.pool_bg(ctx * (1-mask_lr))
        bg_ctx = self.conv_b(bg_pool)
        weight = torch.sigmoid(self.conv_g(torch.cat([fg_ctx, bg_ctx], dim=1))).squeeze(3).squeeze(2)
        sparse_weight = weight[x.indices[:,0].long()]
        # x.features = x.features * sparse_weight
        x = x.replace_feature(x.features * sparse_weight)
        return x


class SparseDecoder3_18(spconv.SparseModule):
    def __init__(self, inc=512):
        super(SparseDecoder3_18, self).__init__()

        # upconv modules
        self.conv_up1 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(inc, 256, kernel_size=3, bias=True, indice_key='spconv2'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )

        self.conv_up2 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(256 + 64, 256, kernel_size=3, bias=True, indice_key='spconv1'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )

        self.conv_up3 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(256 + 64, 64, kernel_size=3, bias=True, indice_key='spconv0'),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )

        chn = 64 + 3

        self.conv_up4_alpha = spconv.SparseSequential(
            spconv.SubMConv2d(chn, 32, kernel_size=3, padding=1, bias=True, indice_key='subm0s'),
            nn.LeakyReLU(),
            spconv.SubMConv2d(32, 16, kernel_size=3, padding=1, bias=True, indice_key='subm0s'),
            nn.LeakyReLU(),
            spconv.SubMConv2d(16, 1, kernel_size=1, padding=0, bias=False, indice_key='subm0s')
        )

        self.conv_p8x = spconv.SubMConv2d(256, 1, kernel_size=1, padding=0, bias=False, indice_key='spconv2')
        self.conv_p4x = spconv.SubMConv2d(256, 1, kernel_size=1, padding=0, bias=False, indice_key='spconv1')
        self.conv_p2x = spconv.SubMConv2d(64, 1, kernel_size=1, padding=0, bias=False, indice_key='spconv0')

    def forward(self, img, conv_out, coarse=None, is_training=True):
        x1, x2, x3, x4, x5 = conv_out

        dec4x = self.conv_up1(x5)
        p4x = self.conv_p8x(dec4x)

        # dec4x.features = torch.cat((dec4x.features, x2.features), 1)
        dec4x = dec4x.replace_feature(torch.cat((dec4x.features, x2.features), 1))
        dec2x = self.conv_up2(dec4x)
        p2x = self.conv_p4x(dec2x)

        # dec2x.features = torch.cat((dec2x.features, x1.features), 1)
        dec2x = dec2x.replace_feature(torch.cat((dec2x.features, x1.features), 1))
        dec1x = self.conv_up3(dec2x)
        p1x = self.conv_p2x(dec1x)

        # img.features = img.features[:,:3] * 0.5 + 0.5
        img = img.replace_feature(img.features[:,:3] * 0.5 + 0.5)
        # dec1x.features = torch.cat((dec1x.features, img.features),1)
        dec1x = dec1x.replace_feature(torch.cat((dec1x.features, img.features),1))
        p0x = self.conv_up4_alpha(dec1x)

        raws = [p4x.dense(), p2x.dense(), p1x.dense(), p0x.dense()]
        # p4x.features = torch.sigmoid(p4x.features)
        p4x = p4x.replace_feature(torch.sigmoid(p4x.features))
        # p2x.features = torch.sigmoid(p2x.features)
        p2x = p2x.replace_feature(torch.sigmoid(p2x.features))
        # p1x.features = torch.sigmoid(p1x.features)
        p1x = p1x.replace_feature(torch.sigmoid(p1x.features))
        # p0x.features = torch.sigmoid(p0x.features)
        p0x = p0x.replace_feature(torch.sigmoid(p0x.features))
        outs = [p4x.dense(), p2x.dense(), p1x.dense(), p0x.dense()]
        return outs


class SHM(nn.Module):
    def __init__(self, inc=4):
        super(SHM, self).__init__()

        self.conv = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        self.ctx = SparseCAM(512, 32, with_norm=True)
        self.backbone = l_sparse_resnet18(inc, stride=8)
        self.decoder = SparseDecoder3_18()

    def forward(self, inputs, lr_pred, coords, batch_size, spatial_shape, ctx):

        x = spconv.SparseConvTensor(inputs, coords.int(), spatial_shape, batch_size)
        encoded_feats = self.backbone(x)
        ctx = self.conv(ctx)
        encoded_feats[-1] = self.ctx(coords.int(), encoded_feats[-1], ctx, lr_pred)
        outs = self.decoder(x, encoded_feats)
        return outs