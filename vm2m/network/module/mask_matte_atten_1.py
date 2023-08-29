import torch
import torch.nn as nn
from torch.nn import functional as F
from vm2m.utils.utils import resizeAnyShape
from .position_encoding import TemporalPositionEmbeddingSine
from .mask_attention import MLP, SelfAttentionLayer, CrossAttentionLayer, FFNLayer

'''
Similar to MaskAttenHead but with trimap queries separately
'''

class MaskMatteAttenHead(nn.Module):
    def __init__(self, input_dim=256, atten_stride=1.0, attention_dim=256, n_block=2, n_heads=4, output_dim=32, return_feat=True, static_queries=0):
        super().__init__()

        self.n_block = n_block
        self.atten_dim = attention_dim
        self.atten_stride = atten_stride
        self.return_feat = return_feat
        self.pe_layer = TemporalPositionEmbeddingSine(attention_dim)
        
        # Linear layer to map input to attention_dim
        self.feat_proj = MLP(input_dim, attention_dim, attention_dim, 1)
        self.token_proj = MLP(input_dim, attention_dim, attention_dim, 1)
        
        self.sa_layers = nn.ModuleList()
        self.token_feat_ca_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.feat_token_ca_layers = nn.ModuleList()

        for _ in range(n_block):
            self.sa_layers.append(
                SelfAttentionLayer(
                    d_model=attention_dim,
                    nhead=n_heads,
                    dropout=0.0,
                    normalize_before=False
                )
            )
            self.token_feat_ca_layers.append(
                CrossAttentionLayer(
                    d_model=attention_dim,
                    nhead=n_heads,
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
            self.feat_token_ca_layers.append(
                CrossAttentionLayer(
                    d_model=attention_dim,
                    nhead=n_heads,
                    dropout=0.0,
                    normalize_before=False
                )
            )
        
        self.final_token_feat_ca = CrossAttentionLayer(
            d_model=attention_dim,
            nhead=n_heads,
            dropout=0.0,
            normalize_before=False
        )
        self.final_mlp = MLP(attention_dim, attention_dim, output_dim, 1)
        self.decoder_norm = nn.LayerNorm(attention_dim)

        self.static_queries = static_queries
        if static_queries > 0:
            self.static_query_feat = nn.Embedding(static_queries, attention_dim)
            self.static_query_embed = nn.Embedding(static_queries, attention_dim)

        self.dilated_kernel = 15
        self.combine_conv = nn.Conv2d(static_queries + 3, 1, kernel_size=1, stride=1, padding=0, bias=False)

        # Convolutions to smooth features
        self.conv = nn.Sequential(
            nn.Conv2d(attention_dim, attention_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(attention_dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(attention_dim, output_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2),
        )
        if return_feat:
            self.conv_out = nn.Conv2d(attention_dim, input_dim, kernel_size=1, stride=1, padding=0, bias=False)
        if self.atten_stride:
            self.ori_feat_proj = nn.Conv2d(input_dim, attention_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def compute_masks(self, mask):
        '''
        Parameters:
        ----------
        mask: b, n_f, n_i, 1, h, w

        Returns:
        -------
        fg, unk, bg masks: b, n_f, n_i, 1, h, w
        '''
        b, n_f, n_i, h, w = mask.shape
        mask = mask.view(b * n_f * n_i, 1, h, w)

        dilated_m = F.max_pool2d(mask, kernel_size=self.dilated_kernel, stride=1, padding=self.dilated_kernel // 2)
        erosion_m = F.max_pool2d(1 - mask, kernel_size=self.dilated_kernel, stride=1, padding=self.dilated_kernel // 2)
        fg_mask = erosion_m
        bg_mask = 1 - dilated_m
        unk_mask = dilated_m - erosion_m

        fg_mask = fg_mask.view(b, n_f, n_i, h, w)
        bg_mask = bg_mask.view(b, n_f, n_i, h, w)
        unk_mask = unk_mask.view(b, n_f, n_i, h, w)

        return fg_mask, unk_mask, bg_mask

    def forward(self, ori_feat, mask):
        '''
        Params:
        -----
        feat: b * n_f, c, h, w
        mask: b, n_f, n_i, h, w

        Returns:
        -------
        matte: b, n_f, n_i, 1, h, w
        '''
        feat = ori_feat

        # Reduce spatial size of feat
        if self.atten_stride > 1.0:
            feat = F.avg_pool2d(feat, kernel_size=self.atten_stride, stride=self.atten_stride, padding=0)
            ori_feat = self.ori_feat_proj(ori_feat)

        # Resize mask to feat size
        scale_factor = feat.shape[-1] * 1.0 / mask.shape[-1] * 1.0
        mask = resizeAnyShape(mask, scale_factor=scale_factor, use_max_pool=True) 

        # Build tokens from mask and feat with MAP + Conv
        b, n_f, n_i, h, w = mask.shape
        # ori_ni = n_i

        fg_m, unk_m, bg_m = self.compute_masks(mask) # (b, n_f, n_i, 1, h, w)

        feat = feat.view(b, n_f, 1, -1, h * w)
        # mask = mask.view(b, n_f, n_i, 1, h * w)
        fg_m = fg_m.view(b, n_f, n_i, 1, h * w)
        unk_m = unk_m.view(b, n_f, n_i, 1, h * w)
        bg_m = bg_m.view(b, n_f, n_i, 1, h * w)

        
        feat_pos = self.pe_layer(b, n_f, h, w, device=feat.device)               # (b, c_atten, n_f, h, w)
        feat_pos = feat_pos.permute(0, 2, 1, 3, 4).reshape(b, n_f, 1, -1, h * w) # (b, n_f, 1, c_atten, h * w)

        # TODO: Separate the tokens to FG, BG, and UNK
        tokens = []
        token_pos = []
        for m in [fg_m, unk_m, bg_m]:
            tok = (feat * m).sum(dim=-1) / (m.sum(dim=-1) + 1e-6) # (b, n_f, n_i, c)
            tok_pos = (feat_pos * m).sum(dim=-1) / (m.sum(dim=-1) + 1e-6) # (b, n_f, n_i, c_atten)
            tokens.append(tok)
            token_pos.append(tok_pos)
        tokens = torch.stack(tokens, dim=3) # (b, n_f, n_i, 3, c)
        tokens = tokens.reshape(b, n_f, n_i * 3, -1)
        token_pos = torch.stack(token_pos, dim=3) # (b, n_f, n_i, 3, c_atten)
        token_pos = token_pos.reshape(b, n_f, n_i * 3, -1)
        

        # Reshape feat to h * w, n_f * b, c
        feat = feat.permute(4, 2, 1, 0, 3).reshape(h * w, n_f * b, -1) # (h * w, n_f * b, c)
        feat_pos = feat_pos.permute(4, 2, 1, 0, 3).reshape(h * w, n_f * b, -1) # (h * w, n_f * b, c_atten)
        
        # FFN to reduce dimension of tokens and feats
        tokens = self.token_proj(tokens) # (b, n_f, n_i * 3, c_atten)
        feat = self.feat_proj(feat) # (h * w, n_f * b, c_atten)

        # Update with static queries
        if self.static_queries > 0:
            # (static_queries, c_atten) --> (b, n_f, n_i, static_queries, c_atten)
            static_tokens = self.static_query_feat.weight[None, None, None].repeat(b, n_f, n_i, 1, 1)
            tokens = tokens.reshape(b, n_f, n_i, 3, -1)
            tokens = torch.cat([tokens, static_tokens], dim=3)
            tokens = tokens.reshape(b, n_f, n_i * (3 + self.static_queries), -1)

            # Same for token_pos
            static_token_pos = self.static_query_embed.weight[None, None, None].repeat(b, n_f, n_i, 1, 1)
            token_pos = token_pos.reshape(b, n_f, n_i, 3, -1)
            token_pos = torch.cat([token_pos, static_token_pos], dim=3)
            token_pos = token_pos.reshape(b, n_f, n_i * (3 + self.static_queries), -1)

        # import pdb; pdb.set_trace()
        n_q = self.static_queries + n_i * 3  if self.static_queries > 0 else n_i * 3

        # Reshape tokens to n_q * n_f, b, c_atten
        tokens = tokens.permute(2, 1, 0, 3).reshape(-1, b, self.atten_dim)
        token_pos = token_pos.permute(2, 1, 0, 3).reshape(-1, b, self.atten_dim)

        # For each iteration:
        for i in range(self.n_block):
            
            # - Self-attention between tokens in the same image (temporal attention, instance-wise attention)
            # Q, KV: n_q * n_f, b, c_atten
            tokens = tokens.reshape(n_q * n_f, b, -1)
            token_pos = token_pos.reshape(n_q * n_f, b, -1)
            tokens = self.sa_layers[i](tokens, 
                                            tgt_mask=None, 
                                            tgt_key_padding_mask=None, 
                                            query_pos=token_pos)
            tokens = tokens.reshape(n_q, n_f * b, -1)
            token_pos = token_pos.reshape(n_q, n_f * b, -1)

            # - tokens to feat attention (spatial attention)
            # Q: n_q, n_f * b, c_atten
            # KV: h * w, n_f * b, c_atten
            tokens = self.token_feat_ca_layers[i](
                tokens, feat,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=feat_pos, query_pos=token_pos
            )

            # - MLP to smooth tokens
            tokens = self.mlp_layers[i](tokens)

            # - feat to token attention
            # Q: h * w, n_f * b, c_atten
            # KV: n_q, n_f * b, c_atten
            feat = self.feat_token_ca_layers[i](
                feat, tokens,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=token_pos, query_pos=feat_pos
            )

        # tokens to features attention
        # Q: n_q, n_f * b, c_atten
        # KV: h * w, n_f * b, c_atten
        tokens = self.final_token_feat_ca(
            tokens, feat,
            memory_mask=None,
            memory_key_padding_mask=None,
            pos=feat_pos, query_pos=token_pos
        )

        # MLP to build kernel
        tokens = self.final_mlp(tokens) # (n_q, n_f * b, c_out)
        tokens = tokens.reshape(n_q, n_f, b, -1)
        tokens = tokens.permute(2, 1, 0, 3) # (b, n_f, n_q, c_out)
        tokens = tokens.reshape(b * n_f, n_q, -1) # (b * n_f, n_q, c_out)
        tokens = self.decoder_norm(tokens)

        # Smooth feature with 2 Convs
        feat = feat.reshape(h, w, n_f, b, -1)
        feat = feat.permute(3, 2, 4, 0, 1)
        feat = feat.reshape(b * n_f, -1, h, w) # (b * n_f, c, h, w)

        # Upsample to original size and combine with ori_feat if stride > 1.0
        if self.atten_stride > 1.0:
            feat = F.interpolate(feat, scale_factor=self.atten_stride, mode='bilinear', align_corners=True)
            feat = ori_feat + feat

        if self.return_feat:
            out_feat = self.conv_out(feat) # (b * n_f, c, h, w)

        feat = self.conv(feat) # (b * n_f, c_out, h, w)
        
        # Dot product feat with kernel to have matte
        output_mask = torch.einsum('bqc,bchw->bqhw', tokens, feat) # (b * n_f, n_q, h, w)
        
        output_mask = output_mask.reshape(b * n_f, n_i, -1, *ori_feat.shape[-2:]) # (b * n_f, n_i, 3 + static_queries, h, w)
        output_mask = output_mask.reshape(b * n_f * n_i, -1, *ori_feat.shape[-2:]) # (b * n_f * n_i, 3 + static_queries, h, w
        output_mask = self.combine_conv(output_mask) # (b * n_f * ni, 1, h, w)
        output_mask = output_mask.reshape(b * n_f, -1, *ori_feat.shape[-2:]) # (b * n_f, n_i, h, w)

        if self.return_feat:
            return output_mask, out_feat
        return output_mask