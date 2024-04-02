import torch
import torch.nn as nn
from torch.nn import functional as F
from vm2m.utils.utils import resizeAnyShape
from .position_encoding import TemporalPositionEmbeddingSine
from .mask_attention import MLP, SelfAttentionLayer, CrossAttentionLayer, FFNLayer

class MaskMatteDynAttenHead(nn.Module):
    def __init__(self, input_dim=256, atten_stride=1.0, attention_dim=256, n_block=2, n_heads=4, 
                 output_dim=32, num_queries=8, query_stride=16, return_feat=True,  perform_pred=True):
        super().__init__()

        assert return_feat or perform_pred, "At least one of return_feat and perform_pred should be True"
        
        self.n_block = n_block
        self.atten_dim = attention_dim
        self.atten_stride = atten_stride
        self.return_feat = return_feat
        self.perform_pred = perform_pred
        self.query_stride = query_stride
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
        
        if perform_pred:
            self.final_token_feat_ca = CrossAttentionLayer(
                d_model=attention_dim,
                nhead=n_heads,
                dropout=0.0,
                normalize_before=False
            )
            self.final_mlp = MLP(attention_dim, attention_dim, output_dim, 1)
            self.decoder_norm = nn.LayerNorm(attention_dim)

        self.num_queries = num_queries
        self.mask_ratio_layer = nn.Linear(1, attention_dim)
        self.query_conv = nn.Conv2d(num_queries, 1, kernel_size=3, stride=1, padding=1, bias=True)

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
        if self.atten_stride > 1.0:
            self.ori_feat_proj = nn.Conv2d(input_dim, attention_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def compute_queries(self, feat, mask):
        
        # Rescale shape to feat size
        scale_factor = feat.shape[-1] * 1.0 / mask.shape[-1] * 1.0
        if scale_factor < 1.0:
            mask = resizeAnyShape(mask, scale_factor=scale_factor, use_max_pool=True) 

        b, n_f, n_i, h, w = mask.shape

        # Compute mask ratio
        mask_ratio = F.avg_pool2d(mask.view(b, n_f * n_i, h, w), kernel_size=self.query_stride, stride=self.query_stride, padding=0)
        mask_ratio = mask_ratio.reshape(b * n_f * n_i, 1, -1)

        k = mask_ratio.shape[-1]

        # Build query feat
        query_feat = feat.reshape(b, n_f, 1, -1, h, w)
        query_feat = F.avg_pool2d(query_feat.view(b * n_f * n_i, -1, h, w), kernel_size=self.query_stride, stride=self.query_stride, padding=0)
        query_feat = query_feat.reshape(b * n_f * n_i, -1, k)

        # Build query pos
        query_pos = self.pe_layer(b, n_f, h, w, device=feat.device)
        query_pos = query_pos.permute(0, 2, 1, 3, 4).reshape(b, n_f, 1, -1, h, w)
        query_pos = query_pos * mask[:, :, :, None]
        query_pos = F.avg_pool2d(query_pos.view(b * n_f * n_i, -1, h, w), kernel_size=self.query_stride, stride=self.query_stride, padding=0)
        query_pos = query_pos.reshape(b * n_f * n_i, -1,k)
        query_pos = query_pos / (mask_ratio + 1e-6)

        # Select topk queries based on ratio
        topk_ratio, topk_ids = torch.topk(mask_ratio, self.num_queries, dim=-1)
        query_feat = torch.gather(query_feat, dim=-1, index=topk_ids.expand(-1, query_feat.shape[-2], self.num_queries))
        query_pos = torch.gather(query_pos, dim=-1, index=topk_ids.expand(-1, query_pos.shape[-2], self.num_queries))

        # Adding mask ratio to query
        mask_ratio_embed = self.mask_ratio_layer(topk_ratio.permute(0, 2, 1)) # b * n_f * n_i, 1, feat_dim
        mask_ratio_embed = mask_ratio_embed.reshape(b, n_f, n_i, self.num_queries, -1)

        query_feat = query_feat.permute(0, 2, 1).reshape(b, n_f, n_i, self.num_queries, -1)
        query_feat = self.token_proj(query_feat) + mask_ratio_embed
        query_pos = query_pos.permute(0, 2, 1).reshape(b, n_f, n_i, self.num_queries, -1)

        return query_feat, query_pos

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

        # Compute queries on original feature
        tokens, token_pos = self.compute_queries(feat, mask)

        # Reduce spatial size of feat
        if self.atten_stride > 1.0:
            feat = F.avg_pool2d(feat, kernel_size=self.atten_stride, stride=self.atten_stride, padding=0)
            ori_feat = self.ori_feat_proj(ori_feat)

        # Build tokens from mask and feat with MAP + Conv
        b, n_f, n_i= mask.shape[:3]
        h, w = feat.shape[-2:]

        feat = feat.view(b, n_f, 1, -1, h * w)
        
        feat_pos = self.pe_layer(b, n_f, h, w, device=feat.device)               # (b, c_atten, n_f, h, w)
        feat_pos = feat_pos.permute(0, 2, 1, 3, 4).reshape(b, n_f, 1, -1, h * w) # (b, n_f, 1, c_atten, h * w)
        
        # Reshape feat to h * w, n_f * b, c
        feat = feat.permute(4, 2, 1, 0, 3).reshape(h * w, n_f * b, -1) # (h * w, n_f * b, c)
        feat_pos = feat_pos.permute(4, 2, 1, 0, 3).reshape(h * w, n_f * b, -1) # (h * w, n_f * b, c_atten)
        
        # FFN to reduce dimension of tokens and feats
        feat = self.feat_proj(feat) # (h * w, n_f * b, c_atten)

        tokens = tokens.reshape(b, n_f, n_i * self.num_queries, -1) # (b, n_f, n_i * num_queries, c_atten)
        token_pos = token_pos.reshape(b, n_f, n_i * self.num_queries, -1) # (b, n_f, n_i * num_queries, c_atten)

        # import pdb; pdb.set_trace()
        n_i = n_i * self.num_queries

        # Reshape tokens to n_i * n_f, b, c_atten
        tokens = tokens.permute(2, 1, 0, 3).reshape(-1, b, self.atten_dim)
        token_pos = token_pos.permute(2, 1, 0, 3).reshape(-1, b, self.atten_dim)

        # For each iteration:
        for i in range(self.n_block):
            
            # - Self-attention between tokens in the same image (temporal attention, instance-wise attention)
            # Q, KV: n_i * n_f, b, c_atten
            tokens = tokens.reshape(n_i * n_f, b, -1)
            token_pos = token_pos.reshape(n_i * n_f, b, -1)
            tokens = self.sa_layers[i](tokens, 
                                            tgt_mask=None, 
                                            tgt_key_padding_mask=None, 
                                            query_pos=token_pos)
            tokens = tokens.reshape(n_i, n_f * b, -1)
            token_pos = token_pos.reshape(n_i, n_f * b, -1)

            # - tokens to feat attention (spatial attention)
            # Q: n_i, n_f * b, c_atten
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
            # KV: n_i, n_f * b, c_atten
            feat = self.feat_token_ca_layers[i](
                feat, tokens,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=token_pos, query_pos=feat_pos
            )

        if self.perform_pred:
        
            # tokens to features attention
            # Q: n_i, n_f * b, c_atten
            # KV: h * w, n_f * b, c_atten
            tokens = self.final_token_feat_ca(
                tokens, feat,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=feat_pos, query_pos=token_pos
            )

        # Smooth feature with 2 Convs
        feat = feat.reshape(h, w, n_f, b, -1)
        feat = feat.permute(3, 2, 4, 0, 1)
        feat = feat.reshape(b * n_f, -1, h, w) # (b * n_f, c, h, w)

        # Upsample to original size and combine with ori_feat if stride > 1.0
        if self.atten_stride > 1.0:
            feat = F.interpolate(feat, scale_factor=self.atten_stride, mode='bilinear', align_corners=True)
            feat = ori_feat + feat

        feat = self.conv(feat) # (b * n_f, c_out, h, w)

        if self.return_feat:
            out_feat = self.conv_out(feat) # (b * n_f, c, h, w)

        if not self.perform_pred:
            return out_feat, tokens

        # MLP to build kernel
        tokens = self.final_mlp(tokens) # (n_i, n_f * b, c_out)
        tokens = tokens.reshape(n_i, n_f, b, -1)
        tokens = tokens.permute(2, 1, 0, 3) # (b, n_f, n_i, c_out)
        tokens = tokens.reshape(b * n_f, n_i, -1) # (b * n_f, n_i, c_out)
        tokens = self.decoder_norm(tokens)

        # Dot product feat with kernel to have matte
        output_mask = torch.einsum('bqc,bchw->bqhw', tokens, feat) # (b * n_f, n_i, h, w)

        output_mask = output_mask.reshape(-1, self.num_queries, *ori_feat.shape[-2:])
        output_mask = self.query_conv(output_mask) # (b * n_f * n_i, 1, h, w)
        output_mask = output_mask.reshape(b * n_f, -1, *ori_feat.shape[-2:])
        
        if self.return_feat:
            return output_mask, out_feat
        return output_mask