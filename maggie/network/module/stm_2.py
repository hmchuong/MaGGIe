import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class STM(nn.Module):
    def __init__(self, in_channel, mask_channel=3, os=16, drop_out=0.0, top_k=4):
        super().__init__()
        self.query_k_conv = nn.Conv2d(in_channel, in_channel // 8, 1, 1, 0, bias=False)
        self.query_v_conv = nn.Conv2d(in_channel, in_channel // 2, 1, 1, 0, bias=False)
        self.query_drop_out = nn.Dropout2d(drop_out)

        embed_mask_channel = 2**(int(math.log2(os))) * mask_channel
        self.mem_k_conv = nn.Conv2d(in_channel + embed_mask_channel, in_channel // 8, 1, 1, 0, bias=False)
        self.mem_v_conv = nn.Conv2d(in_channel + embed_mask_channel, in_channel // 2, 1, 1, 0, bias=False)

        if os > 1:
            self.mask_downsample = nn.Sequential(
                *[nn.Conv2d(mask_channel * (2**i), mask_channel * (2**(i+1)), 3, 2, 1, bias=False) for i in range(int(math.log2(os)))]
            )

        self.smooth_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self.top_k = top_k
    
    def generate_mem(self, feat, mask):
        '''
        Generate memory feature from feat and mask
        feat: b, c , h, w
        mask: b, n_i, h, w
        '''
        with torch.no_grad():
            masks = (mask > 0.5).long()
            mask_ids = torch.arange(1, masks.shape[1]+1, device=masks.device)[None, :, None, None]
            masks = (masks * mask_ids).long()
            mask_embed = self.mask_embed_layer(masks.long())
            mask_embed = mask_embed * (masks > 0).float().unsqueeze(-1)
            mask_embed = mask_embed.sum(1) / ((masks > 0).float().unsqueeze(-1).sum(1) + 1e-6)
            mask_embed = mask_embed.permute(0, 3, 1, 2) # b, c_e, h, w
        mask_embed = self.mask_downsample(mask_embed) # b, c_e_d, h, w
        return torch.cat([feat.detach(), mask_embed], dim=1)

    def forward(self, x, mem):
        '''
        x: b, c, h, w
        prev_feat: b, t, c, h, w
        '''
        b, t, _, h, w = mem.shape

        mem = mem.type(x.dtype)

        # Compute k, v of query
        query_k = self.query_k_conv(x) # b x c // 8 x h x w
        query_v = self.query_v_conv(x) # b x c // 2 x h x w

        # Compute k, v of memory
        mem_k = self.mem_k_conv(mem.flatten(0, 1)) # bt x c // 8 x h x w
        mem_v = self.mem_v_conv(mem.flatten(0, 1)) # bt x c // 2 x h x w

        # Compute attention
        query_k = query_k.flatten(2, 3) # b x c_k x hw
        mem_k = mem_k.view(b, t, -1, h, w) # b x t x c_k x h x w
        mem_k = mem_k.permute(0, 2, 1, 3, 4) # b x c_k x t x h x w
        mem_k = mem_k.flatten(2, 4) # b x c_k x thw
        query_k = F.normalize(query_k, dim=1)
        mem_k = F.normalize(mem_k, dim=1)
        attention = torch.bmm(query_k.transpose(1, 2), mem_k) # b x hw x thw
        attention = F.softmax(attention, dim=2)

        # Compute top k attention to hw
        _, topk_ids =  torch.topk(attention, self.top_k, dim=2)  # b x hw x k

        # Convert flatten ids to 3d ids
        t_ids = torch.div(topk_ids, h*w, rounding_mode='trunc') # b x hw x k
        y_ids = torch.div(topk_ids - t_ids * h*w, w, rounding_mode='trunc') # b x hw x k
        x_ids = topk_ids - (t_ids * h * w + y_ids * w)
        topk_ids = torch.stack([t_ids, y_ids, x_ids], dim=3)

        # Sum up memory value
        mem_v = mem_v.view(b, t, -1, h, w) # b x t x c_v x h x w
        mem_v = mem_v.permute(0, 2, 1, 3, 4) # b x c_v x t x h x w
        mem_v = mem_v.flatten(2, 4) # b x c_v x thw

        mem_v = torch.einsum('bqk,bck->bqc', attention, mem_v) # b x hw x c_v
        mem_v = mem_v.transpose(1, 2) # b x c_v x hw
        mem_v = mem_v.reshape_as(query_v) # b x c_v x h x w

        # Concatenate query and memory values
        # import pdb; pdb.set_trace()
        query_v = self.query_drop_out(query_v)
        query_v = torch.cat([query_v, mem_v], dim=1) # b x c x h x w

        # Smooth
        query_v = self.smooth_conv(query_v)
        
        return query_v, topk_ids

class STM_TopK(STM):
    def forward(self, x, mem, mem_ids):
        '''
        x: b, c, h, w
        mem: b, c, hw, k
        mem_ids: (b, hw, k, 3), triple (t_i, y_i, x_i) of memory values
        '''
        # import pdb; pdb.set_trace()
        b = mem.shape[0]

        mem = mem.type(x.dtype)

        # Compute k, v of query
        query_k = self.query_k_conv(x) # b x c // 8 x h x w
        query_v = self.query_v_conv(x) # b x c // 2 x h x w

        # Compute k, v of memory
        mem_k = self.mem_k_conv(mem) # b x c // 8 x hw x k
        mem_v = self.mem_v_conv(mem) # b x c // 2 x hw x k

        # Compute attention
        query_k = query_k.flatten(2, 3) # b x c_k x hw
        query_k = F.normalize(query_k, dim=1)
        mem_k = F.normalize(mem_k, dim=1)
        # attention = torch.bmm(query_k.transpose(1, 2), mem_k) # b x hw x k
        attention = torch.einsum('bcq,bcqk->bqk', query_k, mem_k) # b x hw x k
        attention = F.softmax(attention, dim=2)

        # Collect new topk_ids
        _, topk_ids = torch.topk(attention, self.top_k, dim=2)  # b x hw x new_k
        b_ids = torch.arange(b)[:, None, None].expand(b, topk_ids.shape[1], self.top_k)
        hw_ids = torch.arange(topk_ids.shape[1])[None, :, None].expand(b, topk_ids.shape[1], self.top_k)

        topk_ids = mem_ids[b_ids.flatten(), hw_ids.flatten(), topk_ids.flatten()] # b x hw x new_k x 3)
        topk_ids = topk_ids.reshape(b, mem_ids.shape[1], self.top_k, 3)
        
        # Sum up memory value
        mem_v = torch.einsum('bqk,bcqk->bcq', attention, mem_v) # b x c_v x hw
        mem_v = mem_v.reshape_as(query_v)

        # Concatenate query and memory values
        query_v = self.query_drop_out(query_v)
        query_v = torch.cat([query_v, mem_v], dim=1) # b x c x h x w

        # Smooth
        query_v = self.smooth_conv(query_v)

        return query_v, topk_ids


