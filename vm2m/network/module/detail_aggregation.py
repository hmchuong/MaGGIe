import math
import torch
from torch import nn
import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp

class DetailAggregation(nn.Module):
    def __init__(self, dim, os=4):
        super().__init__()
        self.aggregation_spconv = spconv.SubMConv2d(dim, dim, 3, stride=1, padding=1, bias=False, dilation=2)
        self.smooth_spconv = spconv.SubMConv2d(dim, dim, 1, stride=1, padding=0, bias=True)

        # downsample
        if os > 1:
            self.downsample = nn.Sequential(
                *[nn.Conv2d(2**i, 2**(i+1), 3, stride=2, padding=1, bias=False) for i in range(int(math.log2(os)))]
            )
        additional_channel = 2 ** int(math.log2(os)) if os > 1 else 0
        self.mem_spconv = spconv.SubMConv2d(dim + additional_channel, dim, 1, stride=1, padding=0, bias=True)
    
    def generate_mem(self, feat, mask):
        
        mask = mask.flatten(0, 1).unsqueeze(1)
        mask = self.downsample(mask)
        
        coords = feat.indices.long()
        mask = mask.permute(0, 2, 3, 1).contiguous()
        mask = mask[coords[:, 0], coords[:, 1], coords[:, 2]]
        feat = feat.replace_feature(torch.cat([feat.features.detach(), mask], dim=1))
        feat = self.mem_spconv(feat)
        return feat
    
    def forward(self, x, mem):
        '''
        x: SparseTensor
        mem: SparseTensor
        '''
        # TODO: Convert mem to target dimension

        
        x_indices = x.indices
        indice_dict = x.indice_dict
        x = Fsp.sparse_add(x, mem)
        x = self.aggregation_spconv(x)

        # Extract indices of original x only
        new_mask = torch.zeros((x.batch_size, *x.spatial_shape), dtype=torch.bool, device=x_indices.device)
        x_indices = x_indices.long()
        new_mask[x_indices[:, 0], x_indices[:, 1], x_indices[:, 2]] = True
        new_indices = x.indices.long()
        valid_indices = torch.nonzero(new_mask[new_indices[:, 0], new_indices[:, 1], new_indices[:, 2]]).squeeze(1)
        x = spconv.SparseConvTensor(x.features[valid_indices], x.indices[valid_indices], x.spatial_shape, x.batch_size, indice_dict=indice_dict)
        x = self.smooth_spconv(x)
        return x