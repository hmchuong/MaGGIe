import torch
import torch.nn as nn
from torch.nn import functional as F
from vm2m.utils.utils import resizeAnyShape
from .position_encoding import TemporalPositionEmbeddingSine
from .mask_attention import MLP, SelfAttentionLayer, CrossAttentionLayer, FFNLayer

class MaskMatteEmbAttenHead(nn.Module):
    def __init__(self, input_dim=256, atten_stride=1.0, attention_dim=256, n_block=2, n_head=4, 
                 output_dim=32, return_feat=True, max_inst=10, use_temp_pe=True, use_id_pe=True, temporal_query='none'):
        super().__init__()
        
        self.n_block = n_block
        self.atten_dim = attention_dim
        self.atten_stride = atten_stride
        self.return_feat = return_feat
        self.max_inst = max_inst
        self.use_id_pe = use_id_pe
        self.pe_layer = TemporalPositionEmbeddingSine(attention_dim)
        
        # Linear layer to map input to attention_dim
        self.feat_proj = MLP(input_dim, attention_dim, attention_dim, 1)
        # self.token_proj = MLP(input_dim, attention_dim, attention_dim, 1)
        
        self.sa_layers = nn.ModuleList()
        self.token_feat_ca_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.feat_token_ca_layers = nn.ModuleList()

        for _ in range(n_block):
            self.sa_layers.append(
                SelfAttentionLayer(
                    d_model=attention_dim,
                    nhead=n_head,
                    dropout=0.0,
                    normalize_before=False
                )
            )
            self.token_feat_ca_layers.append(
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
            self.feat_token_ca_layers.append(
                CrossAttentionLayer(
                    d_model=attention_dim,
                    nhead=n_head,
                    dropout=0.0,
                    normalize_before=False
                )
            )
        
        self.final_token_feat_ca = CrossAttentionLayer(
            d_model=attention_dim,
            nhead=n_head,
            dropout=0.0,
            normalize_before=False
        )
        self.final_mlp = MLP(attention_dim, attention_dim, output_dim, 1)
        self.decoder_norm = nn.LayerNorm(output_dim)

        self.n_temp_embed = self.pe_layer.temporal_num_pos_feats if use_temp_pe else 0
        self.n_id_embed = self.atten_dim - self.n_temp_embed
        self.query_feat = nn.Embedding(max_inst, attention_dim)
        # self.query_embed = nn.Embedding(max_inst, attention_dim)
        self.id_embedding = nn.Embedding(max_inst + 1, self.n_id_embed)
        nn.init.xavier_uniform_(self.id_embedding.weight)
        nn.init.xavier_uniform_(self.query_feat.weight)

        # Convolutions to smooth features
        self.conv = nn.Sequential(
            nn.Conv2d(attention_dim, attention_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(attention_dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(attention_dim, output_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # self.up_conv = nn.Conv2d(attention_dim * 2, attention_dim, kernel_size=1, stride=1, padding=0, bias=False)
        if return_feat:
            self.conv_out = nn.Conv2d(attention_dim, input_dim, kernel_size=1, stride=1, padding=0, bias=False)
            nn.init.xavier_uniform_(self.conv_out.weight)

        if self.atten_stride > 1.0:
            self.ori_feat_proj = nn.Conv2d(input_dim, attention_dim, kernel_size=1, stride=1, padding=0, bias=False)
            nn.init.xavier_uniform_(self.ori_feat_proj.weight)

        self.temporal_query = temporal_query
        if temporal_query and 'mlp' in temporal_query:
            self.mem_mlp = FFNLayer(d_model=attention_dim,
                                    dim_feedforward=attention_dim,
                                    dropout=0.0,
                                    normalize_before=False)
        
        self.temp_layernorm = nn.LayerNorm(attention_dim)

        if temporal_query and 'lstm' in temporal_query:
            self.query_lstm = nn.LSTM(attention_dim, attention_dim, batch_first=True)
            # self.temp_layernorm = nn.LayerNorm(attention_dim)
            for p in self.query_lstm.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def compute_atten_loss(self, b, n_f, guidance_mask, atten_mat):
        # Compute loss on attention mat: max at guidance, min at non-guidance
        # import pdb; pdb.set_trace()

        # Compute max loss
        atten_values = (guidance_mask * atten_mat).sum(2)
        atten_gt = torch.ones_like(atten_values)
        atten_gt[guidance_mask.sum(2) == 0] = 0
        cur_max_loss = (atten_gt - atten_values).sum() / (n_f * b)

        # Compute min loss
        atten_values = ((1.0 - guidance_mask.float()) * atten_mat).sum(2)
        valid_mask = guidance_mask.sum(2) > 0
        cur_min_loss = (atten_values * valid_mask).sum() / (n_f * b)


        # max_values = (guidance_mask * atten_mat).max(2)[0] # n_f * b, n_i
        # cur_max_loss = torch.ones_like(max_values) - max_values
        # cur_max_loss = (guidance_mask.sum(2) > 0).float() * cur_max_loss
        # cur_max_loss = (cur_max_loss.sum()) / (n_f * b)

        # # For each instance, min_mask = all other instance mask, max of min_mask --> 0
        # union_mask = (guidance_mask.sum(1) > 0).float()
        # min_mask = union_mask.unsqueeze(1) - guidance_mask.float()
        # min_values = (min_mask * atten_mat).max(2)[0]
        # cur_min_loss = (min_values.sum()) / (n_f * b)

        return cur_max_loss, cur_min_loss

    def forward(self, ori_feat, mask, prev_tokens=None, use_mask_atten=True, gt_mask=None, aggregate_mem_fn=None):
        '''
        Params:
        -----
        feat: b * n_f, c, h, w
        mask: b, n_f, n_i, h, w
        prev_tokens: b, n_i, c_atten
        Returns:
        -------
        matte: b, n_f, n_i, 1, h, w
        '''
        feat = ori_feat

        # Reduce spatial size of feat
        if self.atten_stride > 1.0:
            feat = F.avg_pool2d(feat, kernel_size=self.atten_stride, stride=self.atten_stride, padding=0)
            ori_feat = self.ori_feat_proj(ori_feat)

        # Build tokens from mask and feat with MAP + Conv
        scale_factor = feat.shape[-1] * 1.0 / mask.shape[-1] * 1.0
        mask = resizeAnyShape(mask, scale_factor=scale_factor, use_max_pool=True) 
        b, n_f= mask.shape[:2]
        h, w = feat.shape[-2:]

        feat = feat.view(b, n_f, 1, -1, h * w)
        
        # Compute n_channels for ID and temporal embedding
        n_temp_embed = self.n_temp_embed
        # n_id_embed = self.n_id_embed

        # Feat pos: ID embedding + Temporal position embedding
        temp_feat_pos = None
        if n_temp_embed > 0:
            temp_feat_pos = self.pe_layer(b, n_f, 1, 1, device=feat.device)               # (b, c_atten, n_f, 1, 1)
            temp_feat_pos = temp_feat_pos.repeat(1, 1, 1, h, w)                                # (b, c_atten, n_f, h, w)
            temp_feat_pos = temp_feat_pos[:, :n_temp_embed] # (b, c_atten_temp, n_f, h, w)

        # Adding ID embedding to feat_pos
        id_feat_pos = torch.arange(1, mask.shape[2]+1, device=mask.device)[None, None, :, None, None] # (1, 1, n_i, 1, 1)
        id_feat_pos = (mask * id_feat_pos).max(2)[0]
        id_feat_pos = self.id_embedding(id_feat_pos.long()) # (b, n_f, h, w, c_atten_id)
        # id_feat_pos = id_feat_pos[:, None].repeat(1, n_id_embed, 1, 1, 1) # (b, c_atten_id, n_f, h, w)
        id_feat_pos = id_feat_pos.permute(0, 4, 1, 2, 3) # (b, c_atten_id, n_f h, w)

        if temp_feat_pos is not None:
            feat_pos = torch.cat([id_feat_pos, temp_feat_pos], dim=1) # (b, c_atten, n_f, h, w)
        else:
            feat_pos = id_feat_pos
        feat_pos = feat_pos.permute(0, 2, 1, 3, 4).reshape(b, n_f, 1, -1, h * w) # (b, n_f, 1, c_atten, h * w)
        

        # Learnable Token feat
        tokens = self.query_feat.weight[None].repeat(b, 1, 1) # (b, max_inst,  c_atten)
        # mask = mask.view(b, n_f, -1, 1, h * w)
        # tokens = (feat * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-6) # (b, n_f, n_i, c)
        # if tokens.shape[2] < self.max_inst:
        #     tokens = F.pad(tokens, (0, 0, self.max_inst - tokens.shape[2], 0), mode='constant', value=0)


        # Token pos: ID embedding + Temporal position embedding
        id_token_pos = torch.arange(1, self.max_inst + 1, device=tokens.device)[None, None, :] # (1, 1, max_inst)
        id_token_pos = self.id_embedding(id_token_pos.long()) # (1, 1, max_inst, c_atten_id)
        id_token_pos = id_token_pos.repeat(b, 1, 1, 1).squeeze(1) # (b, max_inst, c_atten_id)
        if temp_feat_pos is not None:
            temp_token_pos = temp_feat_pos[:, :, :, None, 0, 0].permute(0, 2, 3, 1).repeat(1, 1, self.max_inst, 1) # (b, n_f, max_inst, c_atten_temp)
            token_pos = torch.cat([id_token_pos, temp_token_pos], dim=-1) # (b, n_f, max_inst, c_atten)
        else:
            token_pos = id_token_pos
            
        # import pdb; pdb.set_trace()

        # Reshape feat to h * w, n_f * b, c
        feat = feat.permute(4, 2, 1, 0, 3).reshape(h * w * n_f, b, -1) # (h * w * n_f, b, c)
        feat_pos = feat_pos.permute(4, 2, 1, 0, 3).reshape(h * w * n_f, b, -1) # (h * w * n_f, b, c_atten)

        # TODO: Test the similarity between token_pos and feat_pos
        # mat1 = feat_pos.permute(1, 0, 2) # (n_f * b, h * w, c_atten)
        # mat2 = token_pos.permute(0, 1, 3, 2).reshape(b * n_f, -1, self.max_inst) # (n_f * b, max_inst, c_atten)
        # a_mat = torch.matmul(mat1, mat2) # (n_f * b, h * w, max_inst)
        # a_mat = (a_mat - a_mat.min()) / (a_mat.max() - a_mat.min() + 1e-8)
        # mask_valid_ids = mask.sum((3, 4))
        # import cv2
        # cv2.imwrite("test_mask.png", mask[1, 0, 5].detach().cpu().numpy() * 255)
        # cv2.imwrite("test_amat.png", a_mat[1, :, :, 5].detach().cpu().numpy() * 255)
        # feat_pos[0, 1].T @ toekn_pos[1, 0, 6]
        # import pdb; pdb.set_trace()
        
        # FFN to reduce dimension of tokens and feats
        feat = self.feat_proj(feat) # (h * w, n_f * b, c_atten)
        # tokens = self.token_proj(tokens) # (b, n_f, n_i, c_atten)

        # import pdb; pdb.set_trace()
        n_i = self.max_inst

        # Reshape tokens to n_i, b, c_atten
        tokens = tokens.permute(1, 0, 2).reshape(-1, b, self.atten_dim)
        token_pos = token_pos.permute(1, 0, 2).reshape(-1, b, self.atten_dim)

        # combine with previous tokens
        # if prev_tokens is not None:
            # prev_tokens = prev_tokens.permute(1, 0, 2).unsqueeze(1)  # n_i, 1, b, c_atten
            # prev_tokens = prev_tokens.repeat(1, n_f, 1, 1)  # n_i, n_f, b, c_atten
            # import pdb; pdb.set_trace()
            # tokens = tokens + prev_tokens #.flatten(0, 1)  # n_i * n_f, b, c_atten


        # atten_padding_mask = n_f * b, h * w, n_i
        if self.training:
            gt_mask = resizeAnyShape(gt_mask, scale_factor=scale_factor, use_max_pool=True)  if not use_mask_atten else mask
            atten_padding_m = gt_mask.permute(1, 0, 2, 3, 4)
            atten_padding_m = atten_padding_m.reshape(n_f * b, -1, h * w)
            if atten_padding_m.shape[1] < n_i:
                atten_padding_m = torch.cat([atten_padding_m, torch.zeros((n_f * b, n_i - atten_padding_m.shape[1], h * w), device=atten_padding_m.device)], dim=1)
            atten_padding_m = atten_padding_m > 0
            guidance_mask = atten_padding_m.clone()
            invalid_m = atten_padding_m.sum(-1) == 0
            invalid_m = invalid_m[:, :, None].repeat(1, 1, h * w)
            atten_padding_m[invalid_m] = True
            atten_padding_m = ~atten_padding_m
            
            # For the same instance queries
            atten_padding_m = atten_padding_m.reshape(n_f, b, n_i, -1).permute(1, 2, 3, 0).flatten(2, 3)
            guidance_mask = guidance_mask.reshape(n_f, b, n_i, -1).permute(1, 2, 3, 0).flatten(2, 3)
        
        max_loss = 0
        min_loss = 0
        mem_tokens = None

        # import pdb; pdb.set_trace()
        hidden_state = None

        valid_tokens = mask.sum((1, 3, 4)) > 0
        token_padding_mask = torch.zeros((b, n_i), device=tokens.device).bool()
        if valid_tokens.shape[1] < n_i:
            valid_tokens = torch.cat([valid_tokens, torch.zeros((b, n_i - valid_tokens.shape[1]), device=tokens.device).bool()], dim=1)
        token_padding_mask[~valid_tokens] = True

        # For each iteration:
        for i in range(self.n_block):
            
            # - Self-attention between tokens in the same image (temporal attention, instance-wise attention)
            # Q, KV: n_i * n_f, b, c_atten
            # tokens = tokens.reshape(n_i * n_f, b, -1)
            # token_pos = token_pos.reshape(n_i * n_f, b, -1)
            # tokens = self.sa_layers[i](tokens, 
            #                                 tgt_mask=None, 
            #                                 tgt_key_padding_mask=None, 
            #                                 query_pos=token_pos)
            # tokens = tokens.reshape(n_i, n_f * b, -1)
            # token_pos = token_pos.reshape(n_i, n_f * b, -1)

            # TODO: It's may not good at the first iteration? Moving to the end
            # DEBUG: Single query for instance
            # tokens = tokens.reshape(n_i, b, -1)
            # token_pos = token_pos.reshape(n_i, b, -1)
            # tokens = self.sa_layers[i](tokens, 
            #                                 tgt_mask=None, 
            #                                 tgt_key_padding_mask=None, 
            #                                 query_pos=token_pos)
            # tokens = tokens.reshape(n_i, b, -1)
            # token_pos = token_pos.reshape(n_i, b, -1)

            # - tokens to feat attention (spatial attention)
            # Q: n_i, b, c_atten
            # KV: h * w * n_f, b, c_atten
            # import pdb; pdb.set_trace()
            
            # import pdb; pdb.set_trace()
            tokens, atten_mat = self.token_feat_ca_layers[i](
                tokens, feat,
                memory_mask=atten_padding_m if use_mask_atten else None,
                memory_key_padding_mask=None,
                pos=feat_pos if self.use_id_pe else None, query_pos=token_pos if self.use_id_pe else None
            )
            # attention = atten_mat.view(n_f, n_i, h, w)
            # import cv2
            # cv2.imwrite("attention.png", (attention[2,0].cpu().numpy() > 0.0001) * 255)
            # import pdb; pdb.set_trace()
            if self.training and not use_mask_atten:
                cur_max_loss, cur_min_loss = self.compute_atten_loss(b, n_f, guidance_mask, atten_mat)
                max_loss += cur_max_loss
                min_loss += cur_min_loss

            # atten_mat = atten_mat.reshape(b, n_i, h, w)
            # valid_mask = mask.sum((3, 4)) > 0
            # import cv2
            # cv2.imwrite("test_mask.png", mask[0, 0, 5].detach().cpu().numpy() * 255)
            # vis_atten = atten_mat[0, 0].detach().cpu().numpy()
            # vis_atten = (vis_atten - vis_atten.min()) / (vis_atten.max() - vis_atten.min() + 1e-8)
            # cv2.imwrite("test_atten_mask.png", vis_atten * 255)
            # pos_atten = torch.einsum('nbc,sbc->nsb', token_pos, feat_pos)
            # vis_pos = pos_atten[5, :, 0]
            # vis_pos = (vis_pos - vis_pos.min()) / (vis_pos.max() - vis_pos.min() + 1e-8)
            # vis_pos = vis_pos.reshape(h, w)
            # cv2.imwrite("test_pos_mask.png", vis_pos.detach().cpu().numpy() * 255)
            # import pdb; pdb.set_trace()

            # - MLP to smooth tokens
            tokens = self.mlp_layers[i](tokens)

            # - feat to token attention
            # Q: h * w * n_f, b, c_atten
            # KV: n_i * n_f, b, c_atten
            feat, _ = self.feat_token_ca_layers[i](
                feat, tokens,
                memory_mask=None,
                memory_key_padding_mask=token_padding_mask,
                pos=token_pos if self.use_id_pe else None, query_pos=feat_pos if self.use_id_pe else None
            )

            tokens = self.sa_layers[i](tokens, 
                                            tgt_mask=None, 
                                            tgt_key_padding_mask=token_padding_mask, 
                                            query_pos=token_pos)
            
            # Features propagation here?
            if aggregate_mem_fn is not None:
                feat = feat.reshape(h, w, n_f, b, -1)
                feat, hidden_state = aggregate_mem_fn(x=feat.permute(3, 2, 4, 0, 1))
                # b, n_f, c, h, w --> h, w, n_f, b, c
                feat = feat.permute(3, 4, 1, 0, 2).reshape(h * w * n_f, b, -1)
                feat = self.temp_layernorm(feat)
            # TODO: Add previous tokens here
            # if self.temporal_query and prev_tokens and i == 0:
            #     tokens = tokens + prev_tokens
            #     mem_tokens = tokens

        
        
        # tokens to features attention
        # Q: n_i, b, c_atten
        # KV: h * w, b, c_atten
        tokens, atten_mat = self.final_token_feat_ca(
            tokens, feat,
            memory_mask=atten_padding_m if use_mask_atten else None,
            memory_key_padding_mask=None,
            pos=feat_pos, query_pos=token_pos
        )

        # TODO: Aggregate the memory here
        mem_tokens = None
        if self.temporal_query and self.temporal_query != 'none': # and prev_tokens is not None:
            # tokens = tokens + prev_tokens
            
            if 'lstm' in self.temporal_query:
                if prev_tokens is None:
                    tokens, (h_mem, c_mem) = self.query_lstm(tokens.flatten(0,1)[:, None])
                else:
                    tokens, (h_mem, c_mem) = self.query_lstm(tokens.flatten(0,1)[:, None], prev_tokens) 
                tokens = tokens.reshape(-1, n_f * b, self.atten_dim)
                mem_tokens = (h_mem, c_mem)
            elif prev_tokens is not None:
                tokens = tokens + prev_tokens
            # tokens = self.temp_layernorm(tokens)

            if 'mlp' in self.temporal_query:
                tokens = self.mem_mlp(tokens) 
            if mem_tokens is None:
                mem_tokens = tokens


        
            
        # mem_tokens = tokens

        if self.training and not use_mask_atten:
            cur_max_loss, cur_min_loss = self.compute_atten_loss(b, n_f, guidance_mask, atten_mat)
            max_loss += cur_max_loss
            min_loss += cur_min_loss
        
        max_loss = max_loss / (self.n_block + 1)
        min_loss = min_loss / (self.n_block + 1)

        # Smooth feature with 2 Convs
        feat = feat.reshape(h, w, n_f, b, -1)
        feat = feat.permute(3, 2, 4, 0, 1)
        feat = feat.reshape(b * n_f, -1, h, w) # (b * n_f, c, h, w)

        # Upsample to original size and combine with ori_feat if stride > 1.0
        if self.atten_stride > 1.0:
            feat = F.interpolate(feat, scale_factor=self.atten_stride, mode='bilinear', align_corners=True)
            feat = ori_feat + feat
            # feat = torch.cat([ori_feat, feat], dim=1)
            # feat = self.up_conv(feat)
        

        if self.return_feat:
            out_feat = self.conv_out(feat) # (b * n_f, c, h, w)

        feat = self.conv(feat) # (b * n_f, c_out, h, w)

        
        # MLP to build kernel
        tokens = self.final_mlp(tokens) # (n_i, b, c_out)
        tokens = tokens.reshape(n_i, b, -1)
        tokens = tokens.permute(1, 0, 2) # (b, n_i, c_out)
        tokens = tokens.reshape(b, n_i, -1) # (b, n_i, c_out)
        tokens = self.decoder_norm(tokens)

        # Dot product feat with kernel to have matte
        # import pdb; pdb.set_trace()

        output_mask = torch.einsum('bqc,btchw->btqhw', tokens, feat.reshape(b, n_f, -1, h, w)) # (b, n_f, n_i, h, w)
        output_mask = output_mask.flatten(0, 1)
        
        
        # out_debug = torch.einsum('c,chw->hw', tokens[0,0], feat[1])
        # out_debug = (torch.tanh(out_debug) + 1.0) / 2.0
        # import cv2
        # output_mask = (torch.tanh(output_mask) + 1.0) / 2.0
        # cv2.imwrite("test_mask.png", output_mask[0, 0].detach().cpu().numpy() * 255)
        # import pdb; pdb.set_trace()
        
        if self.return_feat:
            return output_mask, out_feat, hidden_state, mem_tokens, max_loss, min_loss
        return output_mask, mem_tokens, max_loss, min_loss