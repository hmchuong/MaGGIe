import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
# from pudb.remote import set_trace

from .position_encoding import TemporalPositionEmbeddingSine

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        if torch.isnan(tgt).any():
            # import pdb; pdb.set_trace()
            # set_trace()
            raise ValueError("Mask is empty")
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        # if torch.isnan(tgt2).any():
        #     import pdb; pdb.set_trace()
            # raise ValueError("Mask is empty")
        tgt = tgt + self.dropout(tgt2)
        # if torch.isnan(tgt).any():
        #     import pdb; pdb.set_trace()
        tgt = self.norm(tgt)
        # if torch.isnan(tgt).any():
        #     import pdb; pdb.set_trace()
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MaskAttentionDynamicKernel(nn.Module):

    def __init__(self, config):
        super().__init__()

        # positional encoding
        # N_steps = config.hidden_dim // 2
        self.pe_layer = TemporalPositionEmbeddingSine(config.hidden_dim, normalize=True)
        self.hidden_dim = config.hidden_dim

        # define Transformer decoder here
        self.num_heads = config.nheads
        self.num_layers = config.dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=config.hidden_dim,
                    nhead=config.nheads,
                    dropout=0.0,
                    normalize_before=config.pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=config.hidden_dim,
                    nhead=config.nheads,
                    dropout=0.0,
                    normalize_before=config.pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=config.hidden_dim,
                    dim_feedforward=config.dim_feedforward,
                    dropout=0.0,
                    normalize_before=config.pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(config.hidden_dim)

        # learnable query features
        self.query_feat = nn.Embedding(2, config.hidden_dim) # One for incoherent, one for refinement decoder

        # learnable query p.e.
        self.query_embed = nn.Embedding(2, config.hidden_dim) # One for incoherent, one for refinement decoder

        # level embedding (we always use 3 scales)
        self.num_feature_levels = len(config.in_features)
        self.level_embed = nn.Embedding(self.num_feature_levels, config.hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if config.in_channels != config.hidden_dim or config.enforce_input_project:
                self.input_proj.append(nn.Conv2d(config.in_channels, config.hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.incoherence_embedder = MLP(config.hidden_dim, config.hidden_dim, config.out_incoherence, 3)
        self.pixeldecoder_embedder = MLP(config.hidden_dim, config.hidden_dim, config.out_pixeldecoder, 3)

    def build_attention_masks(self, masks, shapes):
        '''
        Args:
        masks: (b, n_frames, n_instances, h, w)
        shapes: [(h, w), ...]
        Returns:
        atten_masks: (b * self.num_heads, 2 * n_instances, n_frames * h * w)
        '''
        b_s, n_f, n_ins, ori_h, ori_w = masks.shape
        atten_masks = []
        for h, w in shapes:
            # ds_mask = F.interpolate(masks, (n_ins, h, w), mode='nearest')
            if ori_h > h:
                mask_stride = ori_h // h
                # start = int(mask_stride // 2)
                # ds_mask = masks[:, :, :, start::mask_stride, start::mask_stride].contiguous()
                ds_mask = F.max_pool2d(masks.reshape(b_s * n_f * n_ins, 1, ori_h, ori_w), mask_stride, mask_stride) \
                                            .reshape(b_s, n_f, n_ins, ori_h // mask_stride, ori_w // mask_stride)
            elif ori_h < h:
                ds_mask = F.interpolate(masks, (n_ins, h, w), mode='nearest')
            else:
                ds_mask = masks
            if ds_mask[0, :, 0, :].sum() == 0:
                # import pdb; pdb.set_trace()
                raise ValueError("Mask is empty")
            atten_mask = ds_mask.view(b_s, n_f, n_ins, h * w).permute(0, 2, 1, 3).reshape(b_s, n_ins, n_f * h * w)
            atten_mask = atten_mask.repeat(self.num_heads, 2, 1)
            atten_masks.append(atten_mask == 0)
        
        return atten_masks

    def forward(self, x, masks):
        '''
        x: features from backbone, a list of multi-scale feature, (b * n_frames, c, h/s, w/s) with s=[32, 16, 8, 4]
        mask: mask for each instance (b * n_frames, n_instances, h/8, w/8)
        '''
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        b_s, n_f, n_ins = masks.shape[:3]

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(b_s, n_f, x[i].shape[-2], x[i].shape[-1], x[i].device).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1) # (n_f * h * w, b, embed_dim)
            src[-1] = src[-1].reshape(b_s, n_f, self.hidden_dim, -1)
            src[-1] = src[-1].permute(1, 3, 0, 2).reshape(-1, b_s, self.hidden_dim) # (n_f * h * w, b, embed_dim)

        
        # Build masks
        size_list = [f.shape[-2:] for f in x]
        atten_masks = self.build_attention_masks(masks, size_list)

        # 1 x (b * n_frames * n_instances) x C
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(n_ins, b_s, 1)
        dynamic_kernels = self.query_feat.weight.unsqueeze(1).repeat(n_ins, b_s, 1)

        # src: (n_frames * h * w, b, embed_dim)
        # query: (2 * n_instances, b, embed_dim)
        # attn_mask: (b * self.num_heads, 2 * instances, n_frames * h * w) - masks for each instance in each frame
         
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            
            # if torch.isnan(dynamic_kernels).any():
                # import pdb; pdb.set_trace()
                # set_trace()
                

            # attention: cross-attention first
            dynamic_kernels = self.transformer_cross_attention_layers[i](
                dynamic_kernels, src[level_index],
                memory_mask=atten_masks[level_index],
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            # if torch.isnan(dynamic_kernels).any():
            #     import pdb; pdb.set_trace()

            dynamic_kernels = self.transformer_self_attention_layers[i](
                dynamic_kernels, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # if torch.isnan(dynamic_kernels).any():
            #     import pdb; pdb.set_trace()

            # FFN
            dynamic_kernels = self.transformer_ffn_layers[i](
                dynamic_kernels
            )
        
        # breakdown the dynamic kernels
        incoherence_kernels = dynamic_kernels[:n_ins, ]
        decoder_kernels = dynamic_kernels[n_ins:, ]

        # if torch.isnan(incoherence_kernels).any():
        #     import pdb; pdb.set_trace()

        # Forward to some FFNs to build outputs
        incoherence_kernels = self.incoherence_embedder(incoherence_kernels).permute(1, 0, 2)
        decoder_kernels = self.pixeldecoder_embedder(decoder_kernels).permute(1, 0, 2)
        # if torch.isnan(incoherence_kernels).any():
        #     import pdb; pdb.set_trace()
        return incoherence_kernels, decoder_kernels # (b, n_ins, kernel_size)