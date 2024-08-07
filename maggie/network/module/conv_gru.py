import torch
from torch import nn

class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 dilation: int=1,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, dilation=dilation, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, dilation=dilation, padding=padding),
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
        
    def propagate_features(self, feat, n_f, prev_h_state=None, temp_method='none'):
        hidden_state = None
        if temp_method == 'none':
            all_x = []
            for j in range(n_f):
                o, hidden_state = self.forward(x=feat[:, j], h=None)
                all_x.append(o)
            feat = torch.stack(all_x, dim=1)
        else:
            feat_forward, hidden_state = self.forward(x=feat, h=prev_h_state)

            if temp_method == 'bi':
                # Backward
                feat_backward, _ = self.forward(x=torch.flip(feat[:, :-1], dims=(1,)), h=hidden_state[:, -1])
                feat_backward = torch.flip(feat_backward, dims=(1,))
                feat = feat_forward
                feat[:, :-1] = (feat_forward[:, :-1] + feat_backward) / 2
            else:
                feat = feat_forward
        
        return feat, hidden_state