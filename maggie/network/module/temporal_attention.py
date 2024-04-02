import torch
import torch.nn as nn

class KernelTemporalAttention(nn.Module):
    ''' Channel-wise attention of all frames
    '''
    def __init__(self, dim, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        '''
        x: (b, n_f, n_i, d)
        '''
        n_f = x.shape[1]
        
        # Build positional encoding
        pos_embed = torch.ones((n_f, ), device=x.device, dtype=torch.float32)
        pos_embed = pos_embed.cumsum(0)
        pos_embed = pos_embed / 10000
        pos_embed = pos_embed.sin()
        
        q_pos = pos_embed[None, :, None, None]
        q_pos = q_pos.expand(x.shape[0], -1, x.shape[2], x.shape[3])

        k_pos = q_pos.repeat(1, n_f, 1, 1).reshape(x.shape[0], n_f, n_f, *x.shape[2:])
        k_pos = k_pos.permute(0, 1, 3, 4, 2)

        q = q_pos
        kv = x.repeat(1, n_f, 1, 1).reshape(x.shape[0], n_f, n_f, *x.shape[2:])
        kv = kv.permute(0, 1, 3, 4, 2)
        k = kv + k_pos
        
        # Attention
        atten = q[..., None] * k
        atten = atten.softmax(dim=-1)
        atten = self.attn_drop(atten)

        v = (kv * atten).sum(dim=-1)

        x = x + self.proj_drop(v)
        x = self.norm(x)
        return x

