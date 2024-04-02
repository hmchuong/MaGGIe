import math
import torch
from torch import nn
from torch.nn import functional as F

class STM(nn.Module):
    def __init__(self, in_channel, os=32, embedd_dim=3):
        super().__init__()
        embedd_dim = 0
        self.query_k_conv = nn.Conv2d(in_channel, in_channel // 8, 1, 1, 0, bias=False)
        self.query_v_conv = nn.Conv2d(in_channel, in_channel // 2, 1, 1, 0, bias=False)

        self.mem_k_conv = nn.Conv2d(in_channel + embedd_dim, in_channel // 8, 1, 1, 0, bias=False)
        self.mem_v_conv = nn.Conv2d(in_channel + embedd_dim, in_channel // 2, 1, 1, 0, bias=False)

        # if os > 1:
        #     self.downscale_mask = nn.Sequential(
        #         *[nn.Conv2d(embedd_dim, embedd_dim, 3, 2, 1, bias=False) for _ in range(int(math.log2(os)))]
        #     )

        self.smooth_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Conv2d):
                        nn.init.xavier_uniform_(mm.weight)
                    elif isinstance(mm, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(mm.weight, 1)
                        nn.init.constant_(mm.bias, 0)

    def forward(self, x, prev_mask):
        b, n_f, c = x.shape[:3]
        # if len(prev_mask.shape) < 5:
        #     prev_mask = prev_mask.reshape(b, -1, *prev_mask.shape[1:])
        # if prev_mask.shape[1] > n_f - 1:
        #     prev_mask = prev_mask[:, :n_f - 1]
        # prev_mask = prev_mask.flatten(0, 1)
        
        # print("Perform memory reading")
        # embedding mask, just reuse the mask embedding layer
        # with torch.no_grad():
        #     masks = (prev_mask > 0.5).long()
        #     mask_ids = torch.arange(1, masks.shape[1]+1, device=masks.device)[None, :, None, None]
        #     masks = (masks * mask_ids).long()
        #     mask_embed = self.mask_embed_layer(masks.long())
        #     mask_embed = mask_embed * (masks > 0).float().unsqueeze(-1)
        #     mask_embed = mask_embed.sum(1) / ((masks > 0).float().unsqueeze(-1).sum(1) + 1e-6)
        #     mask_embed = mask_embed.permute(0, 3, 1, 2) # b * t, c_e, h, w
            
        #     mask_embed = mask_embed.reshape(b, n_f - 1, *mask_embed.shape[1:]) # b, t, c_e, H, W
        
        # Resize to the same size of x
        # mask_embed = self.downscale_mask(mask_embed.flatten(0, 1)) # b, t, c_e, h, w
        # mask_embed = mask_embed.reshape(b, n_f - 1, *mask_embed.shape[1:]) # b, t, c_e, h, w

        # Compute k, v of query
        query_k = self.query_k_conv(x[:, -1]) # b x c // 8 x h x w
        query_v = self.query_v_conv(x[:, -1]) # b x c // 2 x h x w

        # Compute k, v of memory
        mem_k = []
        mem_v = []
        for i in range(n_f - 1):
            # cur_f = torch.cat([x[:, i], mask_embed[:, i]], dim=1)
            cur_f = x[:, i]
            mem_k.append(self.mem_k_conv(cur_f))
            mem_v.append(self.mem_v_conv(cur_f))
        mem_k = torch.stack(mem_k, dim=1) # b x t x c // 8 x h x w
        mem_v = torch.stack(mem_v, dim=1) # b x t x c // 2 x h x w

        # Compute attention
        query_k = query_k.view(b, c // 8, -1) # b x c_k x hw
        mem_k = mem_k.view(b, n_f - 1, c // 8, -1) # b x t x c_k x hw
        mem_k = mem_k.transpose(2, 3) # b x t x hw x c_k
        mem_k = mem_k.flatten(1, 2) # b x thw x c_k
        query_k = F.normalize(query_k, dim=1)
        mem_k = F.normalize(mem_k, dim=2)
        attention = torch.bmm(query_k.transpose(1, 2), mem_k.transpose(1, 2)) # b x hw x thw
        attention = torch.softmax(attention, dim=2) # b x hw x thw

        # Sum up memory value
        mem_v = mem_v.permute(0, 2, 1, 3, 4) # b x c_v x t x h x w
        mem_v = mem_v.flatten(2, 4).transpose(1, 2) # b x c_v x thw
        # mem_v = mem_v[:, None] # b x 1 x c_v x thw
        # attention = attention[:, :, None] # b x hw x 1 x thw
        mem_v = torch.einsum('bqk,bkc->bqc', attention, mem_v) # b x hw x c_v
        # mem_v = (attention * mem_v).sum(dim=3) # b x hw x c_v
        mem_v = mem_v.permute(0, 2, 1) # b x c_v x hw
        mem_v = mem_v.reshape_as(query_v) # b x c_v x h x w

        # Concatenate query and memory values
        query_v = torch.cat([query_v, mem_v], dim=1) # b x c x h x w


        # Smooth
        query_v = self.smooth_conv(query_v)

        x = torch.cat([x[:, :-1], query_v[:, None]], dim=1) # b x n_f x c x h x w
        return x
