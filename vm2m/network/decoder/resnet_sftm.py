import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_dec import BasicBlock, ResShortCut_D_Dec

class ConvTmp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tmp_conv = torch.nn.Conv2d(dim, dim, 1, 1, 0)
        self.cur_conv = torch.nn.Conv2d(dim, dim, 1, 1, 0)
        self.ac = nn.Tanh()

    def forward(self, tmp_f, cur_f):
        tmp = self.ac(self.tmp_conv(tmp_f) + self.cur_conv(cur_f))
        return tmp, tmp

class ResShortCut_SFTM(ResShortCut_D_Dec):
    def __init__(self, block, layers, norm_layer=None, large_kernel=False, late_downsample=False):
        super(ResShortCut_D_Dec, self).__init__(block, layers, norm_layer, large_kernel,
                                                late_downsample=late_downsample)

        # Convolution for temporal module
        self.temp_convs = nn.ModuleList()
        self.temp_convs.append(ConvTmp(512))
        self.temp_convs.append(ConvTmp(256))
        self.temp_convs.append(ConvTmp(128))
        self.temp_convs.append(ConvTmp(64))
        self.temp_convs.append(ConvTmp(32))
        self.temp_convs.append(ConvTmp(32))
    
    def aggregate_feature(self, x, conv, n_f):
        x_shape = x.shape
        x = x.reshape(-1, n_f, x_shape[1], x_shape[2], x_shape[3])
        h_t = x[:, 0, :, :, :]
        new_feats = [h_t]
        for i in range(1, n_f):
            h_t, ft = conv(h_t, x[:, i, :, :, :])
            new_feats.append(ft)
        new_feats = torch.stack(new_feats, dim=1)
        return new_feats.reshape(x_shape)

    def forward(self, x, mid_fea, return_ctx, n_f=8, **kwargs):
        ret = {}
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']

        # Temporal aggregation features
        x = self.aggregate_feature(x, self.temp_convs[0], n_f)
        fea5 = self.aggregate_feature(fea5, self.temp_convs[1], n_f)
        fea4 = self.aggregate_feature(fea4, self.temp_convs[2], n_f)
        fea3 = self.aggregate_feature(fea3, self.temp_convs[3], n_f)
        fea2 = self.aggregate_feature(fea2, self.temp_convs[4], n_f)
        fea1 = self.aggregate_feature(fea1, self.temp_convs[5], n_f)

        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        if return_ctx:
            ret['ctx'] = x
        x_os8 = self.refine_OS8(x)

        x = self.layer3(x) + fea3
        x_os4 = self.refine_OS4(x)

        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x_os1 = self.refine_OS1(x)

        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        
        x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8

        return ret

def _res_shortcut_sftm(block, layers, **kwargs):
    model = ResShortCut_SFTM(block, layers, **kwargs)
    return model

def res_shortcut_sftm_22(**kwargs):
    """Constructs a resnet_encoder_14 model.
    """
    return _res_shortcut_sftm(BasicBlock, [2, 3, 3, 2], **kwargs)