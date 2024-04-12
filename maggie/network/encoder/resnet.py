import os
import logging
import torch
import torch.nn as nn
from   maggie.network.module import SpectralNorm, conv1x1, conv3x3

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SpectralNorm(conv3x3(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = SpectralNorm(conv3x3(planes, planes))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet_D(nn.Module):
    """
    Implement and pre-train on ImageNet with the tricks from
    https://arxiv.org/abs/1812.01187
    without the mix-up part.
    """

    def __init__(self, block, layers, norm_layer=None, late_downsample=False, is_additional_branch=False, mask_channel=0, **kwargs):
        super(ResNet_D, self).__init__()

        self.logger = logging.getLogger("Logger")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32
        self.start_stride = [1, 2, 1, 2] if late_downsample else [2, 1, 2, 1]
        self.conv1 = SpectralNorm(nn.Conv2d(3 + mask_channel, 32, kernel_size=3,
                                            stride=self.start_stride[0], padding=1, bias=False))
        self.conv2 = SpectralNorm(nn.Conv2d(32, self.midplanes, kernel_size=3, stride=self.start_stride[1], padding=1,
                                            bias=False))
        self.conv3 = SpectralNorm(nn.Conv2d(self.midplanes, self.inplanes, kernel_size=3, stride=self.start_stride[2],
                                            padding=1, bias=False))
        self.bn1 = norm_layer(32)
        self.bn2 = norm_layer(self.midplanes)
        self.bn3 = norm_layer(self.inplanes)
        self.activation = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=self.start_stride[3])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if not is_additional_branch:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer_bottleneck = self._make_layer(block, 512, layers[3], stride=2)

        self.out_channels = {
            'os1': 32, 
            'os2': 32, 
            'os4': 64, 
            'os8': 128, 
            'os16': 256, 
            'os32': 512
        }

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight_bar)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

        # self.logger.debug("encoder conv1 weight shape: {}".format(str(self.conv1.module.weight_bar.data.shape)))
        self.conv1.module.weight_bar.data[:,3:,:,:] = 0

        # self.logger.debug(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(2, stride),
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion, stride)),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        out = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.activation(x) # N x 32 x 512 x 512
        out['os1'] = nn.functional.interpolate(x0, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x0)
        x = self.bn2(x)
        x1 = self.activation(x) # N x 32 x 256 x 256
        out['os2'] = x1
        x = self.conv3(x1)
        x = self.bn3(x)
        x2 = self.activation(x) # N x 64 x 128 x 128

        x3 = self.layer1(x2) # N x 64 x 128 x 128
        out['os4'] = x3
        x4 = self.layer2(x3) # N x 128 x 64 x 64
        out['os8'] = x4
        x5 = self.layer3(x4) # N x 256 x 32 x 32
        out['os16'] = x5
        x6 = self.layer_bottleneck(x5) # N x 512 x 16 x 16
        out['os32'] = x6
        return out # 1, 1/2, 1/4, 1/8, 1/16, 1/32

class ResShortCut_D(ResNet_D):

    def __init__(self, block, layers, num_mask=1, norm_layer=None, late_downsample=False, **kwargs):
        super(ResShortCut_D, self).__init__(block, layers, norm_layer, late_downsample=late_downsample, mask_channel=num_mask)
        first_inplane = 3 + num_mask
        self.shortcut_inplane = [first_inplane, self.midplanes, 64, 128, 256]
        self.shortcut_plane = [32, self.midplanes, 64, 128, 256]

        self.shortcut = nn.ModuleList()
        for stage, inplane in enumerate(self.shortcut_inplane):
            self.shortcut.append(self._make_shortcut(inplane, self.shortcut_plane[stage]))

    def _make_shortcut(self, inplane, planes):
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(inplane, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes),
            SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes)
        )

    def forward(self, x, **kwargs):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        x1 = self.activation(out) # N x 32 x 256 x 256
        out = self.conv3(x1)
        out = self.bn3(out)
        out = self.activation(out)

        x2 = self.layer1(out) # N x 64 x 128 x 128
        x3= self.layer2(x2) # N x 128 x 64 x 64
        x4 = self.layer3(x3) # N x 256 x 32 x 32
        out = self.layer_bottleneck(x4) # N x 512 x 16 x 16

        fea1 = self.shortcut[0](x) # input image and mask
        fea2 = self.shortcut[1](x1)
        fea3 = self.shortcut[2](x2)
        fea4 = self.shortcut[3](x3)
        fea5 = self.shortcut[4](x4)

        return out, {'shortcut':(fea1, fea2, fea3, fea4, fea5), 'image':x[:,:3,...], 'backbone_feat': (x2, x3, x4, out)}

class ResMaskEmbedShortCut_D(ResShortCut_D):
    def __init__(self, block, layers, num_mask=1, num_embed=1, norm_layer=None, late_downsample=False, **kwargs):
        super(ResMaskEmbedShortCut_D, self).__init__(block, layers, num_embed, norm_layer, late_downsample=late_downsample, **kwargs)

        self.num_embed = num_embed

        if self.num_embed > 0:
            self.mask_embed_layer = nn.Embedding(num_mask + 1, num_embed)
    
    def forward(self, x, **kwargs):
        inp = x[:, :3, ...]

        if self.num_embed > 0:
            masks = x[:, 3:, ...]
            mask_ids = torch.arange(1, masks.shape[1]+1, device=masks.device)[None, :, None, None]
             
            # masks = (masks * mask_ids).max(1)[0]
            # masks = masks.sum(1) / (masks > 0).sum(1)
            masks = (masks * mask_ids).long()

            mask_embed = self.mask_embed_layer(masks.long())
            mask_embed = mask_embed * (masks > 0).float().unsqueeze(-1)
            mask_embed = mask_embed.sum(1) / ((masks > 0).float().unsqueeze(-1).sum(1) + 1e-6)
            mask_embed = mask_embed.permute(0, 3, 1, 2)
            
            inp = torch.cat([inp, mask_embed], dim=1)

        return super().forward(inp, **kwargs)

def load_pretrain_state_dict():
    path = "pretrain/model_best_resnet34_En_nomixup.pth"
    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        return state_dict
    return None

def res_encoder_29(**kwargs):
    model = ResNet_D(BasicBlock, [3, 4, 4, 2], **kwargs)
    state_dict = load_pretrain_state_dict()
    if state_dict:
        model.load_state_dict(state_dict, strict=False)
    return model

def _res_shortcut_D(block, layers, **kwargs):
    model = ResShortCut_D(block, layers, **kwargs)
    return model

def _res_shortcut_embed_D(block, layers, **kwargs):
    model = ResMaskEmbedShortCut_D(block, layers, **kwargs)
    return model

def res_shortcut_29(**kwargs):
    """Constructs a resnet_encoder_25 model.
    """
    model = _res_shortcut_D(BasicBlock, [3, 4, 4, 2], **kwargs)
    state_dict = load_pretrain_state_dict()
    if state_dict:
        if kwargs['num_mask'] > 0:
            del state_dict['conv1.module.weight_bar']
            del state_dict['conv1.module.weight_v']
        model.load_state_dict(state_dict, strict=False)
    return model

def res_shortcut_embed_29(**kwargs):
    model = _res_shortcut_embed_D(BasicBlock, [3, 4, 4, 2], **kwargs)
    state_dict = load_pretrain_state_dict()
    if state_dict:
        if kwargs['num_mask'] > 0:
            del state_dict['conv1.module.weight_bar']
            del state_dict['conv1.module.weight_v']
        model.load_state_dict(state_dict, strict=False)
    return model

if __name__ == "__main__":
    m = ResNet_D(BasicBlock, [3, 4, 4, 2])
    for m in m.modules():
        print(m._get_name())
