import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowSTM(nn.Module):
    def __init__(self, in_channel, mask_channel=3, os=16, drop_out=0.0, w_size=15):
        super().__init__()
        self.query_k_conv = nn.Conv2d(in_channel, in_channel // 8, 1, 1, 0, bias=False)
        self.query_v_conv = nn.Conv2d(in_channel, in_channel // 2, 1, 1, 0, bias=False)
        self.query_drop_out = nn.Dropout2d(drop_out)

        self.mem_k_conv = nn.Conv2d(in_channel, in_channel // 8, 1, 1, 0, bias=False)
        self.mem_v_conv = nn.Conv2d(in_channel, in_channel // 2, 1, 1, 0, bias=False)

        self.smooth_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.w_size = w_size

    def forward(self, x, mem):
        '''
        x: b, c, h, w
        prev_feat: b, t, c, h, w
        '''
        mem = torch.cat([x[:, None], mem], dim=1)
        
        b, t, _, h, w = mem.shape

        mem = mem.type(x.dtype)

        # Compute k, v of query
        query_k = self.query_k_conv(x) # b x c // 8 x h x w
        query_v = self.query_v_conv(x) # b x c // 2 x h x w

        # Compute k, v of memory
        mem_k = self.mem_k_conv(mem.flatten(0, 1)) # bt x c // 8 x h x w
        mem_v = self.mem_v_conv(mem.flatten(0, 1)) # bt x c // 2 x h x w

        # Window cross attention
        query_k = query_k.view(b, -1, h, w) # b x c_k x h x w
        mem_k = mem_k.view(b, t, -1, h, w) # b x t x c_k x h x w
        
        # Add padding to mem_k and unfold mem_k
        mem_k = F.pad(mem_k, (self.w_size // 2, self.w_size // 2, self.w_size // 2, self.w_size // 2)) # b x t x c_k x (h + w_size) x (w + w_size)
        unfolded_memk = mem_k.unfold(3, self.w_size, 1).unfold(4, self.w_size, 1) # b x t x c_k x h x w x w_size x w_size
        
        # Same thing for mem_v
        mem_v = mem_v.view(b, t, -1, h, w)
        mem_v = F.pad(mem_v, (self.w_size // 2, self.w_size // 2, self.w_size // 2, self.w_size // 2))
        unfolded_memv = mem_v.unfold(3, self.w_size, 1).unfold(4, self.w_size, 1)
        
        attention = torch.einsum('bchw, btchwxy -> bhwtxy', query_k, unfolded_memk) # b x h * w x t x w_size * w_size
        attention = attention.flatten(3,5) 
        attention = F.softmax(attention, dim=2)

        unfolded_memv = unfolded_memv.permute(0, 2, 3, 4, 1, 5, 6)
        unfolded_memv = unfolded_memv.flatten(4, 6)

        mem_v = torch.einsum('bhwk,bchwk->bchw', attention, unfolded_memv) # b x hw x c_v

        # Concatenate query and memory values
        # import pdb; pdb.set_trace()
        query_v = self.query_drop_out(query_v)
        query_v = torch.cat([query_v, mem_v], dim=1) # b x c x h x w

        # Smooth
        query_v = self.smooth_conv(query_v)
        
        return query_v