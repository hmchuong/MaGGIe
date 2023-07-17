import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp
from vm2m.network.ops import SpectralNorm
from timm.models.layers import trunc_normal_
from .position_encoding import TemporalPositionEmbeddingSine

# Temporal attention: Normal cross attention

class LinearSparseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        
        # q:  ()
        # k: N2 x C
        # v: N2 x C
        # m: 

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 2, B, num_heads, N, C // num_heads
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, key=None):
    """3x3 convolution with padding"""
    return spconv.SubMConv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                                padding=dilation, groups=groups, bias=False, dilation=dilation, indice_key=key)

class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, norm_layer=None, middle_indice_key=None, last_indice_key=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.stride = stride
        conv = conv3x3
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if self.stride > 1:
            # self.conv1 = SpectralNorm(
            #     # spconv.SparseInverseConv2d(inplanes, inplanes, kernel_size=3, bias=True, indice_key=middle_indice_key)
            #     spconv.SparseConvTranspose2d(inplanes, inplanes, kernel_size=4, stride=2, padding=1, bias=False, indice_key=middle_indice_key)
            # )
            self.conv1 = spconv.SparseConvTranspose2d(inplanes, inplanes, kernel_size=4, stride=2, padding=1, bias=False, indice_key=middle_indice_key)
        else:
            self.conv1 = conv(inplanes, inplanes, key=middle_indice_key)
        self.bn1 = norm_layer(inplanes)
        self.activation = nn.LeakyReLU(0.2)
        self.conv2 = conv(inplanes, planes, key=last_indice_key)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.activation(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.upsample is not None:
            identity = self.upsample(x)

        out = Fsp.sparse_add(out, identity)
        out = out.replace_feature(self.activation(out.features))

        return out

class TemporalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x, indices):
        '''
        x: N x C, sparse points
        indices: N x 3, indices of sparse points [bs * n_f, h, w]
        ''' 
        # Find indices of temporal neighbors
        # Fill in the mask with valid indices
        # Compute attention matrix
        # Q: N x C
        # K: n_f x C
        # V: n_f x C
        pass

class SparseNNAttention(nn.Module):
    def __init__(self, d_model, k=100, max_queries=3000, attn_drop=0.):
        super().__init__()
        self.k_nearest = k
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = F.relu
        self.scale = d_model ** -0.5
        self.max_queries = max_queries

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, res_feat, q_feat, k_feat, v_feat, q_pos, k_pos):
        '''
        q: N_q x C, sparse query points
        k: N_kv x C, sparse key points
        v: N_kv x C, sparse value points
        q_pos: N_q x 3, indices of query points [bs * n_f, h, w]
        k_pos: N_kv x 3, indices of key points [bs * n_f, h, w]
        '''
        
        # Limit the number of queries by sampling
        # ori_feat = q_feat
        selected_ids = torch.arange(q_feat.shape[0], device=q_feat.device)
        if q_feat.shape[0] > self.max_queries:
            step = int(math.ceil(q_feat.shape[0] * 1.0 / self.max_queries))
            selected_ids = torch.arange(0, q_feat.shape[0], step=step, device=q_feat.device)

        # Find k-nearest key points for each query point
        dist_mat = torch.sqrt(((q_pos[selected_ids, None] - k_pos[None]) ** 2).sum(-1))
        topk_ids = torch.topk(dist_mat, self.k_nearest, dim=1, largest=False)[1]

        # Compute the attention matrix
        print("num queries", q_feat[selected_ids].shape[0])

        attn = ((q_feat[selected_ids, None] * k_feat[topk_ids])).sum(-1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute the output
        x = (attn[:, :, None] * v_feat[topk_ids]).sum(1)
        
        # Assign back to the original feature
        selected_ids = selected_ids[:, None].expand(-1, x.shape[1])
        res_feat.scatter_(0, selected_ids, x)
        
        # TODO: Peform the Sparse Conv to spread the features

        x = res_feat + self.dropout(x)
        x = self.norm(x)
        
        return x

class UpsamplingNearest2d(spconv.SparseModule):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
    
    # def forward(self, x):
        
    #     indices = x.indices.long()
    #     features = x.features
    #     h, w = x.spatial_shape
    #     batch_size = x.batch_size
    #     dummy_mask = torch.zeros((batch_size, 1, h, w), dtype=torch.bool, device=features.device)
    #     dummy_mask[indices[:, 0], :, indices[:, 1], indices[:, 2]] = True
    #     dummy_mask = F.interpolate(dummy_mask.float(), scale_factor=self.scale_factor, mode='nearest')
    #     dummy_mask = dummy_mask.squeeze(1)
    #     upsample_ids = dummy_mask.nonzero()

    #     # Compute distance between upsample_ids and indices
    #     norm_old_indices = indices.float() / torch.tensor([1, h, w], dtype=torch.float, device=indices.device)
    #     norm_new_indices = upsample_ids.float() / torch.tensor([1, h * self.scale_factor, w * self.scale_factor], dtype=torch.float, device=indices.device)
    #     dist_mat = torch.sqrt(((norm_new_indices[:, None] - norm_old_indices[None]) ** 2).sum(-1))
    #     import pdb; pdb.set_trace()
    #     topk_ids = torch.topk(dist_mat, 1, dim=1, largest=False)[1]
    #     features = features[topk_ids[:, 0]]

    #     return spconv.SparseConvTensor(features, upsample_ids.int(), (int(h * self.scale_factor), int(w * self.scale_factor)), batch_size)

    def forward(self, x):
        
        indices = x.indices.long()
        features = x.features
        h, w = x.spatial_shape
        batch_size = x.batch_size
        
        # Assign low-scale coordinates to mask and upscale coordinate map
        dummy_mask = torch.ones((batch_size, 1, h, w), dtype=torch.int64, device=features.device) * -1
        dummy_mask[indices[:, 0], :, indices[:, 1], indices[:, 2]] = torch.arange(indices.shape[0], device=features.device).long()[:, None]
        dummy_mask = F.interpolate(dummy_mask.float(), scale_factor=self.scale_factor, mode='nearest')
        new_ids = (dummy_mask[:, 0] != -1).nonzero()
        map_ids = dummy_mask[new_ids[:, 0], 0, new_ids[:, 1], new_ids[:, 2]].long()
        features = features[map_ids]
        return spconv.SparseConvTensor(features, new_ids.int(), (int(h * self.scale_factor), int(w * self.scale_factor)), batch_size)

        # Compute distance between upsample_ids and indices
        # norm_old_indices = indices.float() / torch.tensor([1, h, w], dtype=torch.float, device=indices.device)
        # norm_new_indices = upsample_ids.float() / torch.tensor([1, h * self.scale_factor, w * self.scale_factor], dtype=torch.float, device=indices.device)
        # dist_mat = torch.sqrt(((norm_new_indices[:, None] - norm_old_indices[None]) ** 2).sum(-1))
        # import pdb; pdb.set_trace()
        # topk_ids = torch.topk(dist_mat, 1, dim=1, largest=False)[1]
        # features = features[topk_ids[:, 0]]

        # return spconv.SparseConvTensor(features, upsample_ids.int(), (int(h * self.scale_factor), int(w * self.scale_factor)), batch_size)


class ProgressiveSMHADec(nn.Module):
    def __init__(self, emb_dim, shc_dims):
        super(ProgressiveSMHADec, self).__init__()
        block = BasicBlock
        layers = [2, 3, 3, 2]
        late_downsample = False
        large_kernel = False

        norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.kernel_size = 5 if large_kernel else 3
        self.inplanes = 512 if layers[0] > 0 else 256
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32

        self.conv1 = spconv.SparseSequential(
            spconv.SparseConvTranspose2d(self.midplanes, 32, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.LeakyReLU(0.2)
        )

        self.layer1 = self._make_layer(block, 256, layers[0], stride=2, middle_indice_key="layer16_32", last_indice_key="layer16_32_1")
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, middle_indice_key="layer8_16", last_indice_key="layer8_16_1")
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, middle_indice_key="layer4_8", last_indice_key="layer4_8_1")
        self.layer4 = self._make_layer(block, self.midplanes, layers[3], stride=2, middle_indice_key="layer2_4", last_indice_key="layer2_4_1")

        self.refine_OS1 = spconv.SparseSequential(
            spconv.SubMConv2d(32, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            nn.LeakyReLU(0.2)
        )

        self.refine_OS4 = spconv.SparseSequential(
            spconv.SubMConv2d(64, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            nn.LeakyReLU(0.2)
        )

        self.refine_OS8 = spconv.SparseSequential(
            spconv.SubMConv2d(128, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            nn.LeakyReLU(0.2)
        )
    
    def _make_layer(self, block, planes, blocks, stride=1, first_indice_key=None, middle_indice_key=None, last_indice_key=None):
        if blocks == 0:
            return spconv.SparseSequential(nn.Identity())
        norm_layer = self._norm_layer
        upsample = None
        if stride == 2:
            upsample = spconv.SparseSequential(
                # nn.UpsamplingNearest2d(scale_factor=2),
                # SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                UpsamplingNearest2d(scale_factor=2),
                # SpectralNorm(
                #     # spconv.SparseInverseConv2d(self.inplanes, planes * block.expansion, kernel_size=3, bias=True, indice_key=middle_indice_key)
                #     spconv.SubMConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, padding=0, bias=False, indice_key=middle_indice_key)
                # ),
                spconv.SubMConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, padding=0, bias=False, indice_key=middle_indice_key),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            upsample = spconv.SparseSequential(
                # SpectralNorm(
                #     spconv.SubMConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, padding=0, bias=False, indice_key=middle_indice_key)
                # ),
                spconv.SubMConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, padding=0, bias=False, indice_key=middle_indice_key),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, upsample, norm_layer, middle_indice_key=middle_indice_key, last_indice_key=last_indice_key)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return spconv.SparseSequential(*layers)
    
    @torch.no_grad()
    def prepare_sparse_input(self, feat, inc_mask):
        h, w = feat.shape[-2:]
        m_h, m_w = inc_mask.shape[-2:]

        target_mask = None
        if m_h > h:
            # Downsample
            os = m_h // h
            target_mask = F.max_pool2d(inc_mask, kernel_size=os, stride=os)
        else:
            # Upsample
            target_mask = F.interpolate(inc_mask, size=(h, w), mode='nearest')
        
        coords = torch.where(target_mask.squeeze(1) > 0)
        
        # TODO: Sample points in training
        if self.training and len(coords[0]) > 1600000:
            ids = torch.randperm(len(coords[0]))[:1600000]
            coords = [i[ids] for i in coords]

        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat[coords]
        coords = torch.stack(coords, dim=1)

        return feat, coords.int()

    def predict(self, x, dec_k):
        '''
        x: N x 32 x h x w, logit features
        dec_k: N x n_i x D, decoder kernels
        '''
        n_i = dec_k.shape[1]
        x = x.reshape(1, -1, *x.shape[-2:]) # (1, N * 32, h, w)
        dec_k = dec_k.reshape(-1, 32, 3, 3) # (N * n_i, 32, 3, 3)

        out = F.conv2d(x, dec_k, stride=1, padding=1, groups=x.shape[1] // 32) # (1, N * n_i, h, w)
        out = out.reshape(-1, n_i,  *x.shape[-2:]) # (N, n_i, h, w)
        return out

    def forward(self, x, mid_fea, inc_mask, coarse_mask, bs, n_f, n_i, dec_kernels):
        '''
        x: N x C x h/32 x w/32, embedding features
        mid_fea: dict of features from backbone
        inc_mask: N x n_i x h/8 x w/8, unknown masks
        coarse_mask: N x h/8 x w/8, coarse masks
        bs, n_f, n_i: int, batch size, number of frames, number of instances
        '''
        sc_feats = list(mid_fea['shortcut'])

        # Break the kernels into 3 parts
        dec_k_os8 = dec_kernels[:, :, :288]
        dec_k_os4 = dec_kernels[:, :, 288:576]
        dec_k_os1 = dec_kernels[:, :, 576:]

        N, _, h, w = x.shape
        
        # Union all instances
        u_inc_mask = torch.max(inc_mask, dim=1, keepdim=True)[0]

        # Prepare the OS32 coords
        x, x_coords = self.prepare_sparse_input(x, u_inc_mask)
        x = spconv.SparseConvTensor(x, x_coords, (h, w), N)

        # Prepare shortcut features
        for i in range(len(sc_feats)):
            feat = sc_feats[i]
            f_h, f_w = feat.shape[-2:]
            feat, coords = self.prepare_sparse_input(feat, u_inc_mask)
            feat = spconv.SparseConvTensor(feat, coords, (f_h, f_w), N)
            sc_feats[i] = feat
        
        # OS 32 -> OS16
        sc_feats = sc_feats[::-1]
        x = self.layer1(x)
        x = Fsp.sparse_add(x, sc_feats[0])

        # OS 16 -> OS8
        x = self.layer2(x)
        x = Fsp.sparse_add(x, sc_feats[1])

        # Predict OS8
        x_os8 = self.refine_OS8(x).dense()
        x_os8 = self.predict(x_os8, dec_k_os8)
        
        # OS 8 -> OS4
        x = self.layer3(x)
        x = Fsp.sparse_add(x, sc_feats[2])

        # Predict OS4
        x_os4 = self.refine_OS4(x).dense()
        x_os4 = self.predict(x_os4, dec_k_os4)
        
        # OS 4 -> OS2
        x = self.layer4(x)
        x = Fsp.sparse_add(x, sc_feats[3])

        # Combine OS2 with OS1
        x = self.conv1(x)
        x = Fsp.sparse_add(x, sc_feats[4])

        # predict OS1
        x_os1 = self.refine_OS1(x).dense()
        x_os1 = self.predict(x_os1, dec_k_os1)
        
        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        
        x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

        # Fuse with coarse_mask
        inc_mask = F.interpolate(inc_mask, scale_factor=8.0, mode='nearest')
        coarse_mask = F.interpolate(coarse_mask, scale_factor=8.0, mode='nearest')
        x_os1 = coarse_mask * (1.0 - inc_mask) + inc_mask * x_os1
        x_os4 = coarse_mask * (1.0 - inc_mask) + inc_mask * x_os4
        x_os8 = coarse_mask * (1.0 - inc_mask) + inc_mask * x_os8

        ret = {}
        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8
        return ret
    
# class ProgressiveSMHADec(nn.Module):
#     def __init__(self, emb_dim, shc_dims):
#         super(ProgressiveSMHADec, self).__init__()
#         shc_dims = shc_dims[::-1]

#         self.atten_dim = 128
#         max_k_neigbors = 128

#         self.q_proj = nn.ModuleList([
#             nn.Linear(x, self.atten_dim) for x in shc_dims
#         ])
#         kv_dims = [emb_dim] + [self.atten_dim] * (len(shc_dims) - 1)
#         self.kv_proj = nn.ModuleList([
#             nn.Linear(x, self.atten_dim) for x in kv_dims
#         ])

#         self.spatial_attn = nn.ModuleList([
#             SparseNNAttention(self.atten_dim, k=max_k_neigbors // (2**i)) for i in range(len(shc_dims))
#         ])

#     @torch.no_grad()
#     def prepare_sparse_input(self, feat, inc_mask, bs, n_f):
#         h, w = feat.shape[-2:]
#         m_h, m_w = inc_mask.shape[-2:]

#         target_mask = None
#         if m_h > h:
#             # Downsample
#             os = m_h // h
#             target_mask = F.max_pool2d(inc_mask, kernel_size=os, stride=os)
#         else:
#             # Upsample
#             target_mask = F.interpolate(inc_mask, size=(h, w), mode='nearest')
        
#         coords = torch.where(target_mask.squeeze(1) > 0)
#         feat = feat.permute(0, 2, 3, 1).contiguous()
#         feat = feat[coords]
        
#         # position features
#         coord_f = TemporalPositionEmbeddingSine(self.atten_dim)(bs, n_f, h, w, feat.device)
#         coord_f = coord_f.permute(0, 2, 3, 4, 1).reshape(bs * n_f, h, w, -1)
#         coord_f = coord_f[coords]

#         # normalize coords to [0, 1]
#         coords = torch.stack(coords, dim=1)
#         coords = coords.float() / torch.tensor([1, h, w]).to(coords.device).float()

#         return feat, coords, coord_f
    
#     def forward_attention(self, idx, low_feat, inc_mask, bs, n_f, f_kv, c_kv, f_pos_kv):
        
#         # TODO: Temporal attention on f_kv

#         # Spatial cross attention
#         f_q, c_q, f_pos_q = self.prepare_sparse_input(low_feat, inc_mask, bs, n_f)
#         f_q = self.q_proj[idx](f_q)
#         f_kv = self.kv_proj[idx](f_kv)
#         f_k = f_kv + f_pos_kv
#         f_kv = self.spatial_attn[idx](f_q, f_q + f_pos_q, f_k, f_kv, c_q, c_kv)

#         return f_kv, c_q, f_pos_q

#     def forward(self, x, mid_fea, inc_mask, coarse_mask, bs, n_f, n_i):
#         '''
#         x: N x C x h/32 x w/32, embedding features
#         mid_fea: dict of features from backbone
#         inc_mask: N x n_i x h/8 x w/8, unknown masks
#         coarse_mask: N x h/8 x w/8, coarse masks
#         bs, n_f, n_i: int, batch size, number of frames, number of instances
#         '''
#         shortcuts = list(mid_fea['shortcut'])
#         shortcuts = shortcuts[::-1]
        
#         # Perform attention
#         f_kv, c_kv, f_pos_kv = self.prepare_sparse_input(x, inc_mask, bs, n_f)
#         for i in range(len(shortcuts)):
            
#             f_kv, c_kv, f_pos_kv = self.forward_attention(i, shortcuts[i], inc_mask, bs, n_f, f_kv, c_kv, f_pos_kv)
#             import pdb; pdb.set_trace()

#         import pdb; pdb.set_trace()


