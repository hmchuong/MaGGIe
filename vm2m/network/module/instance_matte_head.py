import torch
from torch import nn

class InstanceMatteHead(nn.Module):
    def __init__(self, embedding_layer, in_channel, hidden_channel=32, k_out_channel=8):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.num_inst = embedding_layer.weight.shape[0]
        k_in = embedding_layer.weight.shape[1]

        self.kernel_linear = nn.Linear(k_in, k_out_channel, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel + k_out_channel, hidden_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channel, 1, 1, 1, 0)
        )
        
    
    def forward(self, x, masks):
        '''
        x: b x n_f, in_channel, h, w
        masks: b, n_f, n_i, h, w
        '''
        # x = self.conv1(x) # b x n_f, hidden_channel, h, w
        
        inst_k = self.embedding_layer.weight # n_inst, dim
        inst_k = self.kernel_linear(inst_k) # n_inst, k_out_channel

        masks = masks.flatten(0, 1) # b * n_f, n_i, h, w
        ids = torch.nonzero(masks.sum((2, 3)) > 0)
        output_masks = torch.zeros(x.shape[0], self.num_inst, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)

        for b_i, inst_i in ids:
            # Compute output mask
            k_f = inst_k[inst_i][:, None, None].repeat(1, x.shape[2], x.shape[3]) # k_out_channel, h, w
            f = torch.cat([x[b_i], k_f], dim=0).unsqueeze(0) # 1, in_channel + k_out_channel, h, w
            output_masks[b_i, inst_i] += self.conv(f).squeeze()
        return output_masks
