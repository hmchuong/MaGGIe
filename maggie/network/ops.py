import torch
from   torch import nn
from   torch.nn import Parameter
from   torch.autograd import Variable
from   torch.nn import functional as F


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class memory_attention(nn.Module):

    def __init__(self, in_dim):
        super(memory_attention, self).__init__()
        self.chanel_in = in_dim
        self.image_query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2 , kernel_size=1)
        self.image_key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)

        self.trimap_query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.trimap_key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)


        self.softmax_image = nn.Softmax(dim=-1)  #
        self.softmax_trimap = nn.Softmax(dim=-1)  #

        self.gamma_image = nn.Parameter(torch.zeros(1))
        self.gamma_trimap = nn.Parameter(torch.zeros(1))

    def forward(self, mixture,image,trimap):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """


        m_batchsize, C, width, height = mixture.size()
        mixture_value = mixture.view(m_batchsize, -1, width * height)  # B X C X N

        image_query = self.image_query_conv(image).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        image_key = self.image_key_conv(image).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        image_energy = torch.bmm(image_query, image_key)  # transpose check
        image_attention = self.softmax_image(image_energy)  # BX (N) X (N)
        image_out = torch.bmm(mixture_value, image_attention.permute(0, 2, 1))
        image_out = image_out.view(m_batchsize, C, width, height)

        trimap_query = self.image_query_conv(trimap).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        trimap_key = self.image_key_conv(trimap).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        trimap_energy = torch.bmm(trimap_query, trimap_key)  # transpose check
        trimap_attention = self.softmax_trimap(trimap_energy)  # BX (N) X (N)
        trimap_out = torch.bmm(mixture_value, trimap_attention.permute(0, 2, 1))
        trimap_out = trimap_out.view(m_batchsize, C, width, height)


        out = self.gamma_image*image_out + self.gamma_trimap*trimap_out + mixture


        return out

class memory_attention_single(nn.Module):

    def __init__(self, in_dim):
        super(memory_attention_single, self).__init__()
        self.chanel_in = in_dim
        # self.image_query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2 , kernel_size=1)
        # self.image_key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.image_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2 , kernel_size=1)

        # self.trimap_query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        # self.trimap_key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)


        self.softmax_image = nn.Softmax(dim=-1)  #
        # self.softmax_trimap = nn.Softmax(dim=-1)  #

        self.gamma_image = nn.Parameter(torch.zeros(1))
        # self.gamma_trimap = nn.Parameter(torch.zeros(1))

    # def forward(self, mixture,image,trimap):
    def forward(self, image,mixture):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = mixture.size()
        mixture_value = mixture.view(m_batchsize, -1, width * height)  # B X C X N

        image_query = self.image_conv(image).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        image_key = self.image_conv(image).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        image_energy = torch.bmm(image_query, image_key)  # transpose check
        image_attention = self.softmax_image(image_energy)  # BX (N) X (N)
        image_out = torch.bmm(mixture_value, image_attention.permute(0, 2, 1))
        image_out = image_out.view(m_batchsize, C, width, height)

        # out = self.gamma_image*image_out + image
        out = self.gamma_image*image + image_out
        # out = image_out

        return out

class SpectralNorm(nn.Module):
    """
    Based on https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py
    and add _noupdate_u_v() for evaluation
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        dtype = w.dtype
        for _ in range(self.power_iterations):
            
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data.type(dtype)))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data.type(dtype)))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _noupdate_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        # import pdb; pdb.set_trace()
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        # if torch.is_grad_enabled() and self.module.training:
        # if self.module.training:
        #     self._update_u_v()
        # else:
        #     self._noupdate_u_v()
        self._update_u_v()
        return self.module.forward(*args)


class ASPP(nn.Module):
    '''
    based on https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/deeplab.py
    '''
    def __init__(self, in_channel, out_channel, conv=nn.Conv2d, norm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        mid_channel = 256
        dilations = [1, 2, 4, 8]

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(in_channel, mid_channel, kernel_size=1, stride=1, dilation=dilations[0], bias=False)
        self.aspp2 = conv(in_channel, mid_channel, kernel_size=3, stride=1,
                               dilation=dilations[1], padding=dilations[1],
                               bias=False)
        self.aspp3 = conv(in_channel, mid_channel, kernel_size=3, stride=1,
                               dilation=dilations[2], padding=dilations[2],
                               bias=False)
        self.aspp4 = conv(in_channel, mid_channel, kernel_size=3, stride=1,
                               dilation=dilations[3], padding=dilations[3],
                               bias=False)
        self.aspp5 = conv(in_channel, mid_channel, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(mid_channel)
        self.aspp2_bn = norm(mid_channel)
        self.aspp3_bn = norm(mid_channel)
        self.aspp4_bn = norm(mid_channel)
        self.aspp5_bn = norm(mid_channel)
        self.conv2 = conv(mid_channel * 5, out_channel, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = norm(out_channel)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='nearest')(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

class DepthwiseSeparableASPPModule(nn.Module):
    '''
    based on https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/deeplab.py
    '''
    def __init__(self, in_channels, out_channel, conv=nn.Conv2d, norm=nn.BatchNorm2d):
        super(DepthwiseSeparableASPPModule, self).__init__()
        mid_channel = 256
        dilations = [1, 2, 4, 8]

        self.embed_layers = nn.ModuleList([MLP(in_channel, mid_channel) for in_channel in in_channels])

        conv_cfg = None
        norm_cfg = {'type': 'BN', 'requires_grad': True}
        act_cfg = {'type': 'ReLU'}

        concat_channels = len(dilations) * mid_channel
        self.aspp1 = ConvModule(
                        concat_channels, mid_channel, 1, dilation=dilations[0], padding=0,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.aspp2 = DepthwiseSeparableConvModule(
                        concat_channels, mid_channel, 3, dilation=dilations[1], padding=dilations[1],
                        norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.aspp3 = DepthwiseSeparableConvModule(
                        concat_channels, mid_channel, 3, dilation=dilations[2], padding=dilations[2],
                        norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.aspp4 = DepthwiseSeparableConvModule(
                        concat_channels, mid_channel, 3, dilation=dilations[3], padding=dilations[3],
                        norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.bottleneck = ConvModule(concat_channels, out_channel, kernel_size=3, padding=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')
        # os_size = x[0].size()[2:]
        os_size = x[-1].size()[2:]
        _c = {}
        for i in range(len(self.embed_layers)):
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[i](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=False)

        c = torch.cat(list(_c.values()), dim=1)
        
        c1 = self.aspp1(c)
        c2 = self.aspp2(c)
        c3 = self.aspp3(c)
        c4 = self.aspp4(c)

        x = torch.cat((c1, c2, c3, c4), 1)
        x = self.bottleneck(x)

        return x