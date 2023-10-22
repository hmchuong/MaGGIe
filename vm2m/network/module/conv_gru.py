import torch
from torch import nn
import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp

class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )
        
    def forward_single_frame(self, x, h):
        
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h
    
    def forward_time_series(self, x, h):
        o = []
        all_h = []

        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
            all_h.append(h)
        o = torch.stack(o, dim=1)
        all_h = torch.stack(all_h, dim=1)
        return o, all_h
        
    def forward(self, x, h):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)
    
# class SparseConvGRU(nn.Module):
#     def __init__(self,
#                  channels: int,
#                  kernel_size: int = 3,
#                  padding: int = 1):
#         super().__init__()
#         self.channels = channels
#         self.ih = spconv.Sequential(
#             spconv.SubMConv2d(channels * 2, channels * 2, kernel_size, padding=padding),
#             nn.Sigmoid()
#         )
#         self.hh = spconv.Sequential(
#             spconv.SubMConv2d(channels * 2, channels, kernel_size, padding=padding),
#             nn.Tanh()
#         )
        
#     def forward_single_frame(self, x, h):
#         r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
#         c = self.hh(torch.cat([x, r * h], dim=1))
#         h = (1 - z) * h + z * c
#         return h, h
    
#     def forward_time_series(self, x, h):
#         o = []
#         all_h = []
#         for xt in x.unbind(dim=1):
#             ot, h = self.forward_single_frame(xt, h)
#             o.append(ot)
#             all_h.append(h)
#         o = torch.stack(o, dim=1)
#         all_h = torch.stack(all_h, dim=1)
#         return o, all_h
        
#     def forward(self, x, h):
#         if h is None:
#             h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
#                             device=x.device, dtype=x.dtype)
        
#         if x.ndim == 5:
#             return self.forward_time_series(x, h)
#         else:
#             return self.forward_single_frame(x, h)