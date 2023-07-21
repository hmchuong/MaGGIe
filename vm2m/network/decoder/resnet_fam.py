import torch
from torch.nn import functional as F
from vm2m.network.module.fam import FeatureAggregationModule
from .resnet_dec import BasicBlock, ResShortCut_D_Dec

class ResShortCut_D_Dec_FAM(ResShortCut_D_Dec):
    def __init__(self, block, layers, norm_layer=None, large_kernel=False, late_downsample=False):
        super().__init__(block, layers, norm_layer, large_kernel,
                                                late_downsample=late_downsample)

        self.fam = FeatureAggregationModule(128, reduction=1, window=9)
    
    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.layer1.train(False)
            self.layer2.train(False)

    def forward(self, inputs, mid_fea, xb=None, xf=None, mask=None):
        fea1, fea2, fea3, fea4, fea5 = mid_fea
        with torch.no_grad():
            x = self.layer1(inputs) + fea5
            x = self.layer2(x) + fea4
        features = x
        if xb is None:
            attb = None
            attf = None
        else:
            x, attb, attf, mask = self.fam(x, xb, xf, mask)
        
        ret = {}
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
            
        return ret, features, attb, attf, mask

def _res_shortcut_D_dec_fam(block, layers, **kwargs):
    model = ResShortCut_D_Dec_FAM(block, layers, **kwargs)
    return model

def res_shortcut_decoder_fam_22(**kwargs):
    return _res_shortcut_D_dec_fam(BasicBlock, [2, 3, 3, 2], **kwargs)