import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalNN(nn.Module):
    def __init__(self, input_dim, window_size=3):
        super(TemporalNN, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size

        self.output_fc = nn.Sequential(
            nn.Conv2d(input_dim * 3, input_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU()
        )

        # init weights
        for m in self.output_fc:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            

    def get_window_mask(self, h, w):
        x_coords = torch.arange(w).repeat(h, 1)
        x_offset = torch.tensor([[-1, 0, 1]]).repeat(3, 1)

        y_coords = torch.arange(h).repeat(w, 1).transpose(0, 1)
        y_offset = torch.tensor([[-1, 0, 1]]).repeat(3, 1).transpose(0, 1)

        x_offset = x_offset.flatten()
        y_offset = y_offset.flatten()
        x_nn = x_coords.flatten()[:, None] + x_offset[None, :]
        x_nn = x_nn.clamp(0, w - 1)
        y_nn = y_coords.flatten()[:, None] + y_offset[None, :]
        y_nn = y_nn.clamp(0, h - 1)

        idx_nn = y_nn * w + x_nn
        idx_nn = idx_nn.clamp(0, h * w - 1)

        window_mask = torch.ones(h * w, h * w) * 9999
        window_mask = torch.scatter(window_mask, 1, idx_nn, 0)
        # window_mask[torch.arange(h * w), torch.arange(h * w)] = 9999

        return window_mask
    
    def find_nn(self, x, y, window_mask):
        '''
        x: (bs, c, h, w)
        y: (bs, c, h, w)

        return: (bs, c, h, w)
        '''
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        y = y.flatten(2)
        
        # normalize x, y
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=1)

        # (bs, h * w, c) x (bs, c, h * w) -> (bs, h * w, h * w)
        dist = torch.bmm(x, y)
        
        # Mask out region outside the spatial NN region
        dist = dist + window_mask
       
        # find neearst neighbor
        nn_idx = torch.argmin(dist, dim=-1)
        nn = torch.gather(y.permute(0, 2, 1), 1, nn_idx[:, :, None].repeat(1, 1, x.shape[-1]))
        nn = nn.permute(0, 2, 1).reshape(b, c, h, w)
        return nn

    def forward(self, x):
        '''
        x: (bs, 3, c, h, w)
        '''
        h, w = x.shape[-2:]

        # Find NN between t and t+1, t-1
        prev_f = x[:, 0] # (bs, c, h, w)
        next_f = x[:, 1] # (bs, c, h, w)
        cur_f = x[:, 2] # (bs, c, h, w)
        
        # Prepare window mask
        window_mask = self.get_window_mask(h, w)
        window_mask = window_mask.to(x.device).type(x.dtype)

        # Find NN between t and t-1
        nn_prev = self.find_nn(cur_f, prev_f, window_mask)
        nn_next = self.find_nn(cur_f, next_f, window_mask)

        # Concatenate
        out = torch.cat([nn_prev, nn_next, cur_f], dim=1)
        # out = cur_f + nn_prev + nn_next
        out = self.output_fc(out)
        out = torch.stack([prev_f, next_f, out], dim=1)
        return out
        # (bs, h * w, c) x (bs, c, h * w) -> (bs, h * w, h * w)
