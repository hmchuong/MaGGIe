import torch
import torch.nn as nn

from vm2m.network.ops import SpectralNorm

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.stride = stride
        conv = conv3x3
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if self.stride > 1:
            self.conv1 = SpectralNorm(nn.ConvTranspose2d(inplanes, inplanes, kernel_size=4, stride=2, padding=1, bias=False))
        else:
            self.conv1 = SpectralNorm(conv(inplanes, inplanes))
        self.bn1 = norm_layer(inplanes)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = SpectralNorm(conv(inplanes, planes))
        self.bn2 = norm_layer(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.activation(out)

        return out

class FPN(nn.Module):
    def __init__(self, inplanes):
        '''
        inplanes: [512, 256, 128, 64, 32, 32]
        '''
        super().__init__()
        
        self._norm_layer = nn.BatchNorm2d
        
        atten_planes = 128
        breakdown_planes = 32

        # From os32 to os16
        self.layer1 = self._make_layer(BasicBlock, inplanes[0], atten_planes, 2) # convert os32-512 to os16-256
        self.conv1_init = SpectralNorm(conv1x1(inplanes[1], atten_planes)) # convert os8-128 to os8-256
        self.smooth1_atten = SpectralNorm(conv1x1(atten_planes, atten_planes)) # smooth os16-256
        
        # From os16 to os8
        self.layer2 = self._make_layer(BasicBlock, atten_planes, atten_planes, 2) # convert os16-256 to os8-256
        self.conv2_init = SpectralNorm(conv1x1(inplanes[2], atten_planes)) # convert os8-128 to os8-256
        self.smooth2_atten = SpectralNorm(conv1x1(atten_planes, atten_planes)) # smooth os8-256
        self.smooth2_breakdown = SpectralNorm(conv1x1(atten_planes, breakdown_planes)) # smooth os8-256 for breakdown

        # From os8 to os4
        self.layer3 = self._make_layer(BasicBlock, atten_planes, atten_planes, 2) # convert os8-256 to os4-256
        self.conv3_init = SpectralNorm(conv1x1(inplanes[3], atten_planes)) # convert os4-64 to os4-256
        self.smooth3_atten = SpectralNorm(conv1x1(atten_planes, atten_planes)) # smooth os4-256
        self.smooth3_breakdown = SpectralNorm(conv1x1(atten_planes, breakdown_planes)) # smooth os4-256 for breakdown

        # From os4 to os2
        self.layer4 = self._make_layer(BasicBlock, atten_planes, inplanes[4], 2) # convert os4-256 to os2-256

        # From os2 to os1
        self.layer5 = self._make_layer(BasicBlock, inplanes[4], inplanes[5], 2) # convert os2-32 to os1-32
        self.smooth5_breakdown = SpectralNorm(conv1x1(inplanes[5], breakdown_planes)) # smooth os1-32 for breakdown
    
    def _make_layer(self, block, inplane, plane, stride=1):
        norm_layer = self._norm_layer
        upsample = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SpectralNorm(conv1x1(inplane, plane * block.expansion)),
            norm_layer(plane * block.expansion),
        )

        return block(inplane, plane, stride, upsample, norm_layer)
    
    def forward(self, feats):
        '''
        Returns:
        feats for attentions: 1/16, 1/8, 1/4 with 256 channels
        feats for unknown regions: 1/8 with 32 channels.
        feats for breakdown: 1/4 and 1 with 32 channels.
        '''
        x0, x1, x2, x3, x4, x5 = feats  # 1, 1/2, 1/4, 1/8, 1/16, 1/32
        attention_outputs = [] # 1/16, 1/8, 1/4
        mask_outputs = [] # 1/8, 1/4, 1
        
        import pdb; pdb.set_trace()
        x = self.layer1(x5) + self.conv1_init(x4) # 1/ 16: 256
        attention_outputs += [self.smooth1_atten(x)]

        x = self.layer2(x) + self.conv2_init(x3) # 1/8: 256
        attention_outputs += [self.smooth2_atten(x)]
        mask_outputs += [self.smooth2_breakdown(x)]

        x = self.layer3(x) + self.conv3_init(x2) # 1/4: 256
        attention_outputs += [self.smooth3_atten(x)]
        mask_outputs += [self.smooth3_breakdown(x)]

        x = self.layer4(x) + x1 # 1/2: 32
        x = self.layer5(x) + x0 # 1: 32 
        mask_outputs += [self.smooth5_breakdown(x)]

        return attention_outputs, mask_outputs






