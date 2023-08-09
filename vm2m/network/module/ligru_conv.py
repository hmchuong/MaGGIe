import torch
from torch import nn
from torch.nn import functional as F

class LiGRUConv(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.conv1 = self.make_conv_layer(input_dim, 3, True)
        self.conv2 = self.make_conv_layer(input_dim, 3, False)
        self.conv3 = self.make_conv_layer(input_dim, 3, True)
        self.conv4 = self.make_conv_layer(input_dim, 3, False)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                module.bias.data.zero_()
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def make_conv_layer(self, input_dim, kernel_size, is_norm):
        blocks = [
            nn.Conv2d(input_dim, input_dim, kernel_size, 1, kernel_size // 2, bias=True, groups=input_dim),
            nn.Conv2d(input_dim, input_dim, 1, 1, 0, bias=True),
        ]
        if is_norm:
            blocks.append(nn.BatchNorm2d(input_dim))
        return nn.Sequential(*blocks)

    def compute_new_state(self, x, h):
        '''
        x: (b, c, h, w)
        h: (b, c, h, w)
        '''
        z = F.sigmoid(self.conv1(x) + self.conv2(h))
        h_temp = F.relu(self.conv3(x) + self.conv4(h))
        h = z * h + (1 - z) * h_temp
        return h

    def forward(self, x):
        '''
        x: (b, n_f, c, h, w)
        '''
        n_f = x.shape[1]
        h = x[:, 0]
        new_x = [h]
        for i in range(1, n_f):
            h = self.compute_new_state(x[:, i], h)
            new_x.append(h)
        return torch.stack(new_x, dim=1)
