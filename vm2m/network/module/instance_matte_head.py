import torch
from torch import nn

class InstanceMatteHead(nn.Module):
    def __init__(self, embedding_layer, in_channel, hidden_channel=32, k_out_channel=8):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.num_inst = embedding_layer.weight.shape[0]
        k_in = embedding_layer.weight.shape[1]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_channel)
        )
        
        self.kernel_linear = nn.Linear(k_in, k_out_channel, bias=False)
        self.scale_conv = nn.Sequential(
            nn.Conv2d(hidden_channel + k_out_channel, hidden_channel, 1, 1),
            nn.Tanh()
        )
        
        self.refine_head = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channel, 1, 1, 1, 0)
        )
    
    def forward(self, x):
        '''
        b x n_f, in_channel, h, w
        '''
        x = self.conv1(x) # b x n_f, hidden_channel, h, w

        
        inst_k = self.embedding_layer.weight # n_inst, dim
        inst_k = self.kernel_linear(inst_k) # n_inst, k_out_channel

        # Combine x and ins_k to have (b * n_f, hidden_channel + k_out_channel, h, w)
        x = x[:, None].repeat(1, inst_k.shape[0], 1, 1, 1) # b x n_f, n_i, hidden_channel, h, w
        inst_k = inst_k[None, :, :, None, None].repeat(x.shape[0], 1, 1, x.shape[-2], x.shape[-1]) # b x n_f, n_i, k_out_channel, h, w

        scale_f = torch.cat([x, inst_k], dim=2) # b x n_f, n_i, hidden_channel + k_out_channel, h, w
        scale_f = scale_f.reshape(-1, *scale_f.shape[2:]).contiguous()
        scale_f = self.scale_conv(scale_f) # b * n_f * n_i, hidden_channel, h, w
        x = x.reshape(-1, *x.shape[2:])
        x = x * scale_f

        x = self.refine_head(x) # b * n_f * n_i, 1, h, w
        x = x.reshape(-1, self.num_inst, *x.shape[2:]) # b, n_f, n_i, h, w

        return x
