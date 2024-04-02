import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv

from torch.nn import BatchNorm1d
from collections import OrderedDict

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    "3x3 convolution with padding"
    return spconv.SubMConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key)

def conv7x7(in_planes, out_planes, stride=1, indice_key=None):
    "3x3 convolution with padding"
    return spconv.SubMConv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False, indice_key=indice_key)


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, padding=1, 
            first_indice_key=None, middle_indice_key=None, last_indice_key=None):

        super(BasicBlock, self).__init__()
        if stride == 2:
            self.conv1 = spconv.SparseConv2d(inplanes, planes, 3, stride, dilation=dilation, padding=padding, bias=False, indice_key=middle_indice_key)
        else:
            self.conv1 = spconv.SubMConv2d(inplanes, planes, 3, stride, dilation=dilation, padding=padding, bias=False, indice_key=middle_indice_key)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = spconv.SubMConv2d(planes, planes, 3, 1, padding=1, indice_key=last_indice_key)
        self.bn2 = nn.BatchNorm1d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        # out.features = self.relu1(out.features)
        out = out.replace_feature(self.relu1(out.features))
        out = self.conv2(out)
        # out.features = self.bn2(out.features)
        out = out.replace_feature(self.bn2(out.features))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out.replace_feature(self.relu(out.features + residual.features))
        # out.features = out.features + residual.features
        # out.features = self.relu(out.features)
        return out


class Bottleneck(spconv.SparseModule):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, padding=1, 
            first_indice_key=None, middle_indice_key=None, last_indice_key=None):

        super(Bottleneck, self).__init__()
        self.conv1 = spconv.SubMConv2d(inplanes, planes, kernel_size=1, bias=False, indice_key=first_indice_key)
        self.bn1 = nn.BatchNorm1d(planes)
        if stride == 2:
            self.conv2 = spconv.SparseConv2d(planes, planes, 3, stride=stride, dilation=dilation, padding=padding, bias=False, 
                             indice_key=middle_indice_key)
        else:
            self.conv2 = spconv.SubMConv2d(planes, planes, 3, stride=stride, dilation=dilation, padding=padding, bias=False, 
                             indice_key=middle_indice_key)
        self.bn2 = nn.BatchNorm1d(planes, momentum=0.01)
        self.conv3 = spconv.SubMConv2d(planes, planes * 4, kernel_size=1, bias=False, indice_key=last_indice_key)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)
        out.features = self.relu(out.features)

        out = self.conv3(out)
        out.features = self.bn3(out.features)

        if self.downsample is not None:
            residual = self.downsample(x)

        out.features = out.features + residual.features
        out.features = self.relu(out.features)
        return out


class SparseResNet18(spconv.SparseModule):

    def __init__(self, inc, stride, block, layers, num_classes=1000, inplanes=128, conv7x7=False):
        self.inplanes = inplanes
        super(SparseResNet18, self).__init__()

        self.enc_channels = [64, 64, 128, 256, 512]

        self.conv1 = spconv.SubMConv2d(inc, 64, kernel_size=3, padding=1, bias=False, indice_key='subm0s')
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = spconv.SparseConv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False, indice_key='spconv0')
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = spconv.SubMConv2d(64, 64, kernel_size=3, padding=1, bias=False, indice_key='subm0e')
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU(inplace=True)
        # import pdb; pdb.set_trace()
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, dilation=1, padding=1, first_indice_key='subm1s', middle_indice_key='spconv1', last_indice_key='subm1e')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1, padding=1, 
                          first_indice_key='subm2s', middle_indice_key='spconv2', last_indice_key='subm2e')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=int(max(1,stride/8)), dilation=1, padding=1, 
                          first_indice_key='subm3s', middle_indice_key='spconv3', last_indice_key='subm3e')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=int(max(1,stride/16)), dilation=2, padding=2, 
                          first_indice_key='subm4s', middle_indice_key='spconv4', last_indice_key='subm4e')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, padding=1, 
                        first_indice_key=None, middle_indice_key=None, last_indice_key=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 2:
                downsample = spconv.SparseSequential(
                    spconv.SparseConv2d(self.inplanes, planes * block.expansion, kernel_size=3, stride=stride, padding=1, 
                        bias=False, indice_key=middle_indice_key),
                    nn.BatchNorm1d(planes * block.expansion),
                )
            else:
                downsample = spconv.SparseSequential(
                    spconv.SubMConv2d(self.inplanes, planes * block.expansion, kernel_size=3, stride=stride, padding=1, 
                        bias=False, indice_key=middle_indice_key),
                    nn.BatchNorm1d(planes * block.expansion),
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation, padding, 
            first_indice_key=first_indice_key, middle_indice_key=middle_indice_key, last_indice_key=last_indice_key))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return spconv.SparseSequential(*layers)

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = x.replace_feature(self.relu1(self.bn1(x.features)))
        x = self.conv2(x)
        x = x.replace_feature(self.relu2(self.bn2(x.features)))
        x = self.conv3(x)
        x = x.replace_feature(self.relu3(self.bn3(x.features)))
        outs.append(x)

        x = self.layer1(x)
        outs.append(x)

        x = self.layer2(x)
        outs.append(x)

        x = self.layer3(x)
        outs.append(x)

        x = self.layer4(x)
        outs.append(x)
        return outs


class SparseResNet(spconv.SparseModule):

    def __init__(self, inc, stride, block, layers, num_classes=1000, inplanes=128, conv7x7=False):
        self.inplanes = inplanes
        super(SparseResNet, self).__init__()

        self.enc_channels = [128, 256, 512, 1024, 2048]

        self.conv1 = spconv.SubMConv2d(inc, 64, kernel_size=3, padding=1, bias=False, indice_key='subm0s')
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = spconv.SparseConv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False, indice_key='spconv0')
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = spconv.SubMConv2d(64, 128, kernel_size=3, padding=1, bias=False, indice_key='subm0e')
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, dilation=1, padding=1, 
                          first_indice_key='subm1s', middle_indice_key='spconv1', last_indice_key='subm1e')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1, padding=1, 
                          first_indice_key='subm2s', middle_indice_key='spconv2', last_indice_key='subm2e')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=int(max(1,stride/8)), dilation=1, padding=1, 
                          first_indice_key='subm3s', middle_indice_key='spconv3', last_indice_key='subm3e')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=int(max(1,stride/16)), dilation=2, padding=2, 
                          first_indice_key='subm4s', middle_indice_key='spconv4', last_indice_key='subm4e')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, padding=1, 
                        first_indice_key=None, middle_indice_key=None, last_indice_key=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 2:
                downsample = spconv.SparseSequential(
                    spconv.SparseConv2d(self.inplanes, planes * block.expansion, kernel_size=3, stride=stride, padding=1, 
                        bias=False, indice_key=middle_indice_key),
                    nn.BatchNorm1d(planes * block.expansion),
                )
            else:
                downsample = spconv.SparseSequential(
                    spconv.SubMConv2d(self.inplanes, planes * block.expansion, kernel_size=3, stride=stride, padding=1, 
                        bias=False, indice_key=middle_indice_key),
                    nn.BatchNorm1d(planes * block.expansion),
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation, padding, 
            first_indice_key=first_indice_key, middle_indice_key=middle_indice_key, last_indice_key=last_indice_key))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return spconv.SparseSequential(*layers)

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x.features = self.relu1(self.bn1(x.features))
        x = self.conv2(x)
        x.features = self.relu2(self.bn2(x.features))
        x = self.conv3(x)
        x.features = self.relu3(self.bn3(x.features))
        outs.append(x)

        import pdb; pdb.set_trace()
        x = self.layer1(x)
        outs.append(x)

        x = self.layer2(x)
        outs.append(x)

        x = self.layer3(x)
        outs.append(x)

        x = self.layer4(x)
        outs.append(x)
        return outs


def l_sparse_resnet18(inc, stride=8, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SparseResNet18(inc, stride, BasicBlock, [2, 2, 2, 2], inplanes=64, conv7x7=True)
    if pretrained:
        state_dict = torch.load('pretrained_model/resnet18.pth')
        model.load_state_dict(state_dict, strict=True)
    return model


def l_sparse_resnet50(inc, stride=8, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SparseResNet(inc, stride, Bottleneck, [3, 4, 6, 3], inplanes=128, conv7x7=False)
    if pretrained:
        state_dict = torch.load('pretrained_model/resnet50_v1c.pth')
        model.load_state_dict(state_dict, strict=True)
    return model