import torch
from torch import nn
from torch.nn import functional as F
from vm2m.network.ops import SpectralNorm
from .resnet_dec import BasicBlock
from vm2m.network.module.base import conv1x1

class ResShortCut_DK_D_Dec(nn.Module):

    def __init__(self, block, layers, norm_layer=None, large_kernel=False, late_downsample=False, final_channel=8, pred_n_kernel=8, pred_n_layer=3):
        super(ResShortCut_DK_D_Dec, self).__init__()
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

        self.final_channel = final_channel
        self.pred_n_kernel = pred_n_kernel
        self.pred_n_layer = pred_n_layer

        ## 1 scale
        self.refine_OS1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(32, final_channel, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(final_channel),
            self.leaky_relu,
        )
        
        ## 1/4 scale
        self.refine_OS4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(32, final_channel, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(final_channel),
            self.leaky_relu,
        )
        
        ## 1/8 scale
        self.refine_OS8 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(32, final_channel, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(final_channel),
            self.leaky_relu,
        )

        weight_nums, bias_nums = [], []
        for l in range(self.pred_n_layer):
            if l == 0:
                weight_nums.append(self.final_channel * self.pred_n_kernel)
                bias_nums.append(self.pred_n_kernel)
            elif l == self.pred_n_layer - 1:
                weight_nums.append(self.final_channel)
                bias_nums.append(1)
            else:
                weight_nums.append(self.final_channel * self.pred_n_kernel)
                bias_nums.append(self.final_channel)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight_bar"):
                    nn.init.xavier_uniform_(m.weight_bar)
                else:
                    nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
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

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts)

        return weight_splits, bias_splits

    # def predict(self, x, dec_k):
    #     '''
    #     x: N x 32 x h x w, logit features
    #     dec_k: N x n_i x D, decoder kernels
    #     '''
    #     n_i = dec_k.shape[1]
    #     x = x.reshape(1, -1, *x.shape[-2:]) # (1, N * 32, h, w)
    #     dec_k = dec_k.reshape(-1, 32, 3, 3) # (N * n_i, 32, 3, 3)

    #     out = F.conv2d(x, dec_k, stride=1, padding=1, groups=x.shape[1] // 32) # (1, N * n_i, h, w)
    #     out = out.reshape(-1, n_i,  *x.shape[-2:]) # (N, n_i, h, w)
    #     return out
    
    def predict(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        x = x.reshape(1, -1, *x.shape[-2:])
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x
    
    def forward_head(self, x, params):
        ws, bs = self.parse_dynamic_params(params, self.final_channel, self.weight_nums, self.bias_nums)
        b, _, h, w = x.shape
        x = self.predict(x, ws, bs, params.shape[0])
        x = x.reshape(b, -1, h, w)
        return x
    
    def forward(self, x, mid_fea, dec_kernels, **kwargs):
        ret = {}
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        # Break the kernels into 3 parts
        # dec_k_os8 = dec_kernels[:, :, :288]
        # dec_k_os4 = dec_kernels[:, :, 288:576]
        # dec_k_os1 = dec_kernels[:, :, 576:]
        n_params = dec_kernels.shape[-1] // 3
        dec_kernels = dec_kernels.reshape(-1, n_params * 3)
        dec_k_os8, dec_k_os4, dec_k_os1 = torch.split_with_sizes(dec_kernels, [n_params] * 3, dim=-1)
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        x_os8 = self.refine_OS8(x)
        x_os8 = self.forward_head(x_os8, dec_k_os8)

        x = self.layer3(x) + fea3
        x_os4 = self.refine_OS4(x)
        x_os4 = self.forward_head(x_os4, dec_k_os4)

        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x_os1 = self.refine_OS1(x)
        x_os1 = self.forward_head(x_os1, dec_k_os1)

        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        
        x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8

        return ret

def _res_shortcut_DK_D_dec(block, layers, **kwargs):
    model = ResShortCut_DK_D_Dec(block, layers, **kwargs)
    return model

def res_shortcut_dk_decoder_22(**kwargs):
    """Constructs a resnet_encoder_14 model.
    """
    return _res_shortcut_DK_D_dec(BasicBlock, [2, 3, 3, 2], **kwargs)