import torch
import torch.nn as nn
from torch.nn import functional as F
from vm2m.utils.utils import resizeAnyShape
from .position_encoding import TemporalPositionEmbeddingSine
from .mask_attention import MLP, FFNLayer


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, value,
                     memory_mask,
                     memory_key_padding_mask,
                     pos,
                     query_pos):
        tgt2, atten_mat = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=value, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        # if torch.isnan(tgt2).any():
        #     import pdb; pdb.set_trace()
            # raise ValueError("Mask is empty")
        tgt = tgt + self.dropout(tgt2)
        # if torch.isnan(tgt).any():
        #     import pdb; pdb.set_trace()
        tgt = self.norm(tgt)
        # if torch.isnan(tgt).any():
        #     import pdb; pdb.set_trace()
        return tgt, atten_mat
    
class MaskIDAttention(nn.Module):
    def __init__(self, input_dim=256, attention_dim=256, n_block=2, n_head=4, max_inst=10):
        super().__init__()
        
        self.n_block = n_block
        self.atten_dim = attention_dim
        self.max_inst = max_inst
        self.n_head = n_head
        self.pe_layer = TemporalPositionEmbeddingSine(attention_dim)
        
        # Linear layer to map input to attention_dim
        # self.query_proj = MLP(input_dim, attention_dim, attention_dim, 1)
        # self.key_proj = MLP(input_dim, attention_dim, attention_dim, 1)
        self.input_conv = nn.Conv2d(input_dim, attention_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.input_conv.weight)
        
        self.ca_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()

        for _ in range(n_block):
            self.ca_layers.append(
                CrossAttentionLayer(
                    d_model=attention_dim,
                    nhead=n_head,
                    dropout=0.0,
                    normalize_before=False
                )
            )
            self.mlp_layers.append(
                FFNLayer(
                    d_model=attention_dim,
                    dim_feedforward=attention_dim,
                    dropout=0.0,
                    normalize_before=False
                )
            )
        self.final_mlp = MLP(attention_dim, attention_dim, input_dim, 1)
        self.id_embedding = nn.Embedding(max_inst, attention_dim)
        nn.init.xavier_uniform_(self.id_embedding.weight)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, feat, mask):
        '''
        Params:
        -----
        feat: b * n_f, c, h, w
        mask: b, n_f, n_i, h, w

        Returns:
        -------
        matte: b, n_f, n_i, 1, h, w
        '''

       # Resize mask to feat size
        scale_factor = feat.shape[-1] * 1.0 / mask.shape[-1] * 1.0
        mask = resizeAnyShape(mask, scale_factor=scale_factor, use_max_pool=True) 

        # Build tokens from mask and feat with MAP + Conv
        b, n_f, n_i, h, w = mask.shape
        # ori_ni = n_i

        if n_i < self.max_inst:
            mask = torch.cat([mask, torch.zeros(b, n_f, self.max_inst - n_i, h, w).to(mask.device)], dim=2)

        query = self.input_conv(feat) # (b, c_atten, h, w)
        query = query.view(b, n_f, 1, -1, h * w)
        mask = mask.view(b, n_f, self.max_inst, 1, h * w)
        
        query_pe = self.pe_layer(b, n_f, h, w, device=feat.device)               # (b, c_atten, n_f, h, w)
        query_pe = query_pe.permute(0, 2, 1, 3, 4).reshape(b, n_f, 1, -1, h * w) # (b, n_f, 1, c_atten, h * w)

        key = (query * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-6) # (b, n_f, n_i, c)
        key_pe = (query_pe * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-6) # (b, n_f, n_i, c_atten)
        

        # Reshape feat to h * w, n_f * b, c
        query = query.permute(4, 2, 1, 0, 3).reshape(h * w, n_f * b, -1) # (h * w, n_f * b, c)
        query_pe = query_pe.permute(4, 2, 1, 0, 3).reshape(h * w, n_f * b, -1) # (h * w, n_f * b, c_atten)
        
        # FFN to reduce dimension of tokens and feats
        # key = self.key_proj(key) # (b, n_f, n_i, c_atten)
        # query = self.query_proj(query) # (h * w, n_f * b, c_atten)

        # Q: h * w, n_f * b, c_atten
        # K: n_i, n_f * b, c_atten
        # V: n_i, n_f * b, c_atten

        key = key.permute(2, 1, 0, 3).reshape(self.max_inst, n_f * b, -1) # (max_inst, n_f * b, c_atten)
        key_pe = key_pe.permute(2, 1, 0, 3).reshape(self.max_inst, n_f * b, -1) # (max_inst, n_f * b, c_atten)
        value = self.id_embedding.weight[None, None]
        # import pdb; pdb.set_trace()
        # conf_mat = value[0,0] @ value[0,0].T
        # conf_mat = (conf_mat - conf_mat.min(-1, keepdim=True)[0]) / (conf_mat.max(-1, keepdim=True)[0] - conf_mat.min(-1, keepdim=True)[0])
        value = value.repeat(b, n_f, 1, 1) # (b, n_f, max_inst, c_atten)
        
        value = value.permute(2, 1, 0, 3).reshape(self.max_inst, n_f * b, -1) # (max_inst, n_f * b, c_atten)
        
        mask = mask.permute(4, 2, 1, 0, 3).reshape(h * w, self.max_inst, n_f * b, 1) # (h * w, max_inst, n_f * b, 1)

        memory_mask = mask.permute(2, 3, 0, 1) # (n_f * b, 1, h * w, max_inst)
        memory_mask = memory_mask.repeat(1, self.n_head, 1, 1) # (n_f * b, n_head, h * w, max_inst)
        memory_mask = memory_mask.reshape(n_f * b * self.n_head, h * w, self.max_inst) # (n_f * b * n_head, h * w, max_inst)
        # memory_mask = memory_mask.float()
        # memory_mask = 1.0 - memory_mask.clamp(min=0.05, max=1.0)
        memory_mask = memory_mask.bool()
        memory_mask = ~memory_mask
        memory_mask[torch.where(memory_mask.sum(-1) == memory_mask.shape[-1])] = False
        # import pdb; pdb.set_trace()

        # For each iteration:
        for i in range(self.n_block):
            # Passing memory_mask: (n_f * b, 4, h * w, max_inst)
            query, atten_m = self.ca_layers[i](
                query, key, value,
                memory_mask=memory_mask,
                memory_key_padding_mask=None,
                pos=key_pe, query_pos=query_pe
            )
            # import cv2
            # for k, area in enumerate(mask.sum(0)[:,0]):
            #     if area > 0:
            #         cv2.imwrite("debug_mask.png", mask[:, k, 0, 0].reshape(h, w).detach().cpu().numpy() * 255)
            #         m = atten_m[0, :, k].reshape(h, w).detach().cpu().numpy()
            #         cv2.imwrite("atten_map.png", m / m.max() * 255)
            #         import pdb; pdb.set_trace()
            # - MLP to smooth tokens
            query = self.mlp_layers[i](query)

            # Recompute key
            key = (query[:, None] * mask).sum(0) / (mask.sum(0) + 1e-6) # (max_inst, n_f * b, c_atten)
        
        query = self.final_mlp(query)
        query = query.reshape(h, w, n_f, b, -1)
        query = query.permute(3, 2, 4, 0, 1)
        query = query.reshape(b * n_f, -1, h, w)
        return query
    
        feat = query + feat
        return feat