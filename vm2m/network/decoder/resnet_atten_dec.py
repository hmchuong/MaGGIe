import torch
from torch import nn
from torch.nn import functional as F
from vm2m.network.ops import SpectralNorm
from .resnet_dec import BasicBlock
from vm2m.network.module.base import conv1x1
from vm2m.network.module.mask_matte_atten import MaskMatteAttenHead

class ResShortCut_Atten_Dec(nn.Module):
    def __init__(self, block, layers, norm_layer=None, large_kernel=False, 
                 late_downsample=False, final_channel=32,
                 atten_dims=[32, 64, 128], atten_blocks=[2, 2, 2], 
                 atten_heads=[1, 2, 4], atten_strides=[2, 1, 1]):
        super(ResShortCut_Atten_Dec, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.large_kernel = large_kernel
        self.kernel_size = 5 if self.large_kernel else 3

        self.inplanes = 512 if layers[0] > 0 else 256
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32

        self.conv1 = SpectralNorm(nn.ConvTranspose2d(self.midplanes, 32, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn1 = norm_layer(32)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.tanh = nn.Tanh()
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.midplanes, layers[3], stride=2)

        ## OS1
        self.refine_OS1 = MaskMatteAttenHead(
            input_dim=32,
            atten_stride=atten_strides[0],
            attention_dim=atten_dims[0],
            n_block=atten_blocks[0],
            n_heads=atten_heads[0],
            output_dim=final_channel,
            return_feat=False
        )
        
        ## OS4
        self.refine_OS4 = MaskMatteAttenHead(
            input_dim=64,
            atten_stride=atten_strides[1],
            attention_dim=atten_dims[1],
            n_block=atten_blocks[1],
            n_heads=atten_heads[1],
            output_dim=final_channel
        )
        
        ## 1/8 scale
        self.refine_OS8 = MaskMatteAttenHead(
            input_dim=128,
            atten_stride=atten_strides[2],
            attention_dim=atten_dims[2],
            n_block=atten_blocks[2],
            n_heads=atten_heads[2],
            output_dim=final_channel
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight_bar"):
                    nn.init.xavier_uniform_(m.weight_bar)
                else:
                    nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, upsample, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, mid_fea, b, n_f, n_i, masks, **kwargs):
        '''
        masks: [b * n_f * n_i, 1, H, W]
        '''

        # Reshape masks
        masks = masks.reshape(b, n_f, n_i, masks.shape[2], masks.shape[3])

        ret = {}
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        
        # import pdb; pdb.set_trace() 
        x_os8, x = self.refine_OS8(x, masks) # 200MB

        x = self.layer3(x) + fea3
        x_os4, x = self.refine_OS4(x, masks) # 258MB

        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x_os1 = self.refine_OS1(x, masks) # 1.140GB, 7.776GB

        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        
        x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8

        return ret

def res_shortcut_attention_decoder_22(**kwargs):
    return ResShortCut_Atten_Dec(BasicBlock, [2, 3, 3, 2], **kwargs)