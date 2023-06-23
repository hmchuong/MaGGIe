import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from kornia.losses import binary_focal_loss_with_logits
import spconv.pytorch as spconv

from .module.fpn import FPN
from .module.mask_attention import MaskAttentionDynamicKernel
from .module.position_encoding import TemporalPositionEmbeddingSine
from .module.pixel_encoder import TransformerEncoder, TransformerEncoderLayer
from .loss import GradientLoss, LapLoss, RMSELoss

def conv1x1bnrelu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class VM2M(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        
        # Backbone module
        self.backbone = backbone

        # FPN decoder
        self.fpn = FPN(self.backbone.out_channels)

        # Convolutions to convert to attention layers inputs
        # attention_channel = cfg.dynamic_kernel.in_channels
        # self.atten_features = cfg.dynamic_kernel.in_features
        # self.conv_atten = nn.ModuleList([])
        # for feat_idx in cfg.dynamic_kernel.in_features:
        #     out_channel = self.backbone.out_channels[feat_idx]
        #     self.conv_atten.append(conv1x1bnrelu(out_channel, attention_channel))

        # Breakdown incoherence
        # self.breakdown_channels = cfg.breakdown.in_channels
        # self.breakdown_features = cfg.breakdown.in_features
        # self.conv_breakdown = nn.ModuleList([])
        # for i, feat_idx in enumerate(cfg.breakdown.in_features):
        #     out_channel = self.backbone.out_channels[feat_idx]
        #     self.conv_breakdown.append(conv1x1bnrelu(out_channel, self.breakdown_channels[i]))

        # Attention module to build dynamic kernels
        self.dynamic_kernels_generator = MaskAttentionDynamicKernel(cfg.dynamic_kernel)

        # Laplacian kernels
        self.laplacian_kernel = torch.tensor(
                [-1, -1, -1, -1, 8, -1, -1, -1, -1],
                dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False)

        # Positional encoding
        # self.pe_layers = []
        # for scale_idx, n_channels in enumerate(cfg.breakdown.in_channels):
        #     self.pe_layers.append(TemporalPositionEmbeddingSine(n_channels, normalize=True, scale=scale_idx * 2 * math.pi))

        self.refinement_train_points = cfg.refinement.n_train_points
        self.refinement_test_points = cfg.refinement.n_test_points
        
        # self.combine_feats = nn.Sequential(
        #     nn.Conv2d(sum(self.breakdown_channels), sum(self.breakdown_channels), kernel_size=3, stride=1, padding=1, bias=False, groups=sum(self.breakdown_channels)), # Spatial conv
        #     nn.Conv2d(sum(self.breakdown_channels), 32, kernel_size=1, stride=1, padding=0, bias=False), # Channel conv
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        # )

        self.trans_pred_conv = nn.Sequential(
                nn.Conv2d(257, 128, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, groups=128),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )

        self.pixel_decoder = spconv.SparseSequential(
            spconv.SubMConv2d(65, 32, kernel_size=3, padding=1, bias=True, indice_key='subm1'),
            nn.LeakyReLU(),
            spconv.SubMConv2d(32, 16, kernel_size=3, padding=1, bias=True, indice_key='subm1'),
            nn.LeakyReLU(),
            spconv.SubMConv2d(16, 1, kernel_size=1, padding=0, bias=False, indice_key='subm1'),
            nn.Sigmoid()
        )

        # Pixel encoder
        # encoder_layer = TransformerEncoderLayer(d_model=32, nhead=2)
        # self.pixel_encoder = TransformerEncoder(encoder_layer, num_layers=3)
        
        self.hr_conv = nn.Conv2d(backbone.out_channels['os4'] + backbone.out_channels['os1'], 64, kernel_size=1, stride=1, padding=0, bias=False)

        # Losses
        self.gradient_loss = GradientLoss()
        self.lap_loss = LapLoss()
        self.alpha_loss = nn.L1Loss(reduction='none') if cfg.loss_alpha_type == 'l1' else RMSELoss()
        self.loss_alpha_w = cfg.loss_alpha_w
        self.loss_alpha_grad_w = cfg.loss_alpha_grad_w
        self.loss_alpha_lap_w = cfg.loss_alpha_lap_w
        
        need_init_weights = [self.dynamic_kernels_generator, self.trans_pred_conv, self.pixel_decoder, self.hr_conv, self.fpn]

        # Init weights
        for module in need_init_weights:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def extract_features(self, feats, feature_lists, conv_lists):
        results = []
        for idx, conv in zip(feature_lists, conv_lists):
            results.append(conv(feats[idx]))
        return results
    
    def _detect_boundary(self, boundary_targets):
        self.laplacian_kernel = self.laplacian_kernel.to(boundary_targets.device)
        boundary_targets = F.conv2d(boundary_targets, self.laplacian_kernel, dilation =3, padding=3)
        boundary_targets[boundary_targets > 0.5] = 1.
        boundary_targets[boundary_targets <= -0.5] = 1.
        return boundary_targets

    def detect_spatial_sparsity(self, masks):
        
        b_s, n_f, n_i, h, w = masks.shape
        masks = masks.reshape(b_s*n_f*n_i, 1, h, w).float()

        # boundary_targets: the boundary locations to refine using laplacian ~ dilate + erose
        boundary_targets = self._detect_boundary(masks)
        boundary_targets = self._detect_boundary(boundary_targets)
        
        boundary_targets = boundary_targets.reshape(b_s, n_f, n_i, h, w)
        return boundary_targets

    def detect_temporal_sparsity(self, masks):
        # masks_src: b, n_f, n_i, h, w
        # masks_tgt: b, n_f, n_i, h, w # frame0 + frame0 to frame n - 1
        masks_tgt = torch.cat([masks[:, :1], masks[:, :-1]], dim=1)
        temporal_sparsity = torch.abs(masks - masks_tgt)

        return temporal_sparsity

    # def parse_inc_dynamic_params(self, params, weight_nums, bias_nums):
    #     '''
    #     params: b, n_instances, n_params
    #     weight_nums: list of weight numbers for each layer
    #     bias_nums: list of bias numbers for each layer 
    #     '''

    #     assert params.dim() == 3
    #     assert len(weight_nums) == len(bias_nums)
    #     assert params.size(2) == sum(weight_nums) + sum(bias_nums)
        

    #     num_layers = len(weight_nums)
    #     bs, num_insts = params.shape[:2]
    #     params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=2))
    #     weight_splits = params_splits[:len(weight_nums)]
    #     bias_splits = params_splits[len(weight_nums):]
        
    #     for l in range(num_layers):
    #         weight_splits[l] = weight_splits[l].reshape(bs, 1, num_insts, -1, 1, 1)
    #         bias_splits[l] = bias_splits[l].reshape(bs, 1, num_insts, -1, 1, 1)

    #     return weight_splits, bias_splits

    def parse_pixdec_dynamic_params(self, params, weight_nums, bias_nums):
        '''
        params: b, n_instances, n_params
        weight_nums: list of weight numbers for each layer, e.g. [32 * 8, 8 * 4, 4 * 1]
        bias_nums: list of bias numbers for each layer, e.g. [8, 4, 1]

        returns:
        weights_splits: list of weights for each layer, e.g. [[b * n_instances, 32, 8]
        '''
        assert params.dim() == 3
        assert len(weight_nums) == len(bias_nums)
        assert params.size(2) == sum(weight_nums) + sum(bias_nums)

        num_layers = len(weight_nums)
        bs, num_insts = params.shape[:2]
        params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=2))
        weight_splits = params_splits[:len(weight_nums)]
        bias_splits = params_splits[len(weight_nums):]

        for l in range(num_layers):
            weight_splits[l] = weight_splits[l].reshape(bs * num_insts, -1, bias_nums[l])
            bias_splits[l] = bias_splits[l].reshape(bs * num_insts, 1, bias_nums[l])

        return weight_splits, bias_splits

    def parse_inc_dynamic_params(self, params, weight_nums, bias_nums):
        '''
        params: b, n_instances, n_params (479)
        weight_nums: list of weight numbers for each layer, e.g. [[(32, 4), (4, 4), (4, 1)], [(33, 4), (4, 4), (4, 1)], [(33, 4), (4, 4), (4, 1)]
        bias_nums: list of bias numbers for each layer, e.g. [[4, 4, 1], [4, 4, 1], [4, 4, 1]]

        returns:
        weight_splits: list of weights for each layer, e.g. [[[b * n_instances, 32, 4]]]
        bias_splits: list of biases for each layer e.g. [[[b * n_instances, 4]]]
        '''
        start = 0
        weight_splits = []
        bias_splits = []
        bs, n_ins = params.shape[:2]
        for layer in range(len(weight_nums)):
            layer_weight_splits = []
            layer_bias_splits = []
            for weight_info, bias_info in zip(weight_nums[layer], bias_nums[layer]):
                in_w = weight_info[0]
                out_w = weight_info[1]
                layer_weight_splits.append(params[:, :, start:start + in_w * out_w].reshape(bs, n_ins, in_w, out_w))
                layer_bias_splits.append(params[:, :, start + in_w * out_w:start + in_w * out_w + bias_info].unsqueeze(-2))
                start += in_w * out_w + bias_info
            weight_splits.append(layer_weight_splits)
            bias_splits.append(layer_bias_splits)
        return weight_splits, bias_splits

    def upsample_bin_map(self, bin_map, scale=2):
        bin_map = torch.repeat_interleave(bin_map, scale, dim=-1)
        bin_map = torch.repeat_interleave(bin_map, scale, dim=-2)
        return bin_map

    def downsample_bin_map(self, bin_map, scale=2):
        bin_shape = bin_map.shape[:-2]
        h, w = bin_map.shape[-2:]
        bin_map = F.max_pool2d(bin_map.reshape(np.prod(bin_shape), 1, h, w), kernel_size=scale, stride=scale)
        bin_map = bin_map.reshape((*bin_shape, h // scale, w // scale))
        return bin_map
    
    def forward_trans_pred(self, breakdown_feats, inc_kernels, input_masks=None):

        # Splits weights and biases from incocerence_kernels
        inc_weights, inc_biases = self.parse_inc_dynamic_params(inc_kernels, [[(32, 4), (4,4), (4, 1)]], [[4, 4, 1]])

        # Predict transition regions with dynamic kernels and features
        bs, n_f = breakdown_feats[0].shape[:2]
        n_inst = inc_kernels.shape[1]
        h, w = breakdown_feats[0].shape[-2:]
        x = breakdown_feats[0].expand(-1, -1, n_inst, -1, -1, -1)
        x = torch.cat([x, input_masks.unsqueeze(3).float()], dim=3)

        x = x.reshape(bs * n_f * n_inst, -1, h, w)
        x = self.trans_pred_conv(x)
        x = x.reshape(bs * n_f * n_inst, -1, h * w).permute(0, 2, 1)

        for i, (wi, b) in enumerate(zip(inc_weights[0], inc_biases[0])):
            # Convert w,b to b * n_f * n_inst, c1, c2
            wi = wi.unsqueeze(1).expand(-1, n_f, -1, -1, -1).reshape((bs * n_f * n_inst, wi.shape[-2], wi.shape[-1]))
            b = b.unsqueeze(1).expand(-1, n_f, -1, -1, -1).reshape((bs * n_f * n_inst, b.shape[-2], b.shape[-1]))
            x = torch.bmm(x, wi) + b
            if i < len(inc_weights[0]) - 1:
                x = F.relu(x)
            
        # Convert back to (b, n_f, n_inst, h, w)
        x = x.permute(0, 2, 1) #.reshape((bs, n_f, n_inst, feat.shape[-2], feat.shape[-1]))
        x = x.reshape((bs, n_f, n_inst, h, w))
        
        return x

    def combine_breakdown_feats(self, breakdown_features):
        shapes = [x.shape[-2:] for x in breakdown_features]
        bs, n_f, n_inst = breakdown_features[0].shape[:3]
        xs = []
        for feat in breakdown_features:
            x = feat.reshape(bs * n_f * n_inst, -1, feat.shape[-2], feat.shape[-1])
            x = F.interpolate(x, size=max(shapes), mode='bilinear', align_corners=False)
            xs += [x]
        xs = torch.cat(xs, dim=1)
        xs = self.combine_feats(xs)
        for i in range(len(breakdown_features)):
            kernel_size = max(shapes)[0] // shapes[i][0]
            if kernel_size > 1:
                breakdown_features[i] = F.avg_pool2d(xs, kernel_size=kernel_size, stride=kernel_size)
            else:
                breakdown_features[i] = xs
            
            breakdown_features[i] = breakdown_features[i].reshape(bs, n_f, n_inst, *breakdown_features[i].shape[-3:])

    
    def inc_prediction(self, inc_kernels, masks, breakdown_feats):
        '''
        Args:
        incoherence_kernels: b, n_instances, kernel_size
        masks: (b, n_frames, n_instances, h//8, w//8), coarse masks
        breakdown_feats: (b, c, h//s, w//s), s = 8, 4, 1

        Returns:
        transition_preds: (b, n_frames, n_instances, h//s, w//s) for being supervised during training
        inc_feats: points to refine (b, n_instances, c, n_points)
        inc_pe: positional encoding of points to refine (b, n_instances, c, n_points)
        inc_indices: index of points in masks (b, n_instances, n_points)
        '''
        bs, n_f, n_ints, h, w = masks.shape
        # Compute spatial incoherence
        spat_inc = self.detect_spatial_sparsity(masks)

        # Compute temporal incoherence
        temp_inc = self.detect_temporal_sparsity(masks)
        
        breakdown_feats = [x.reshape(bs, n_f, 1, -1, x.shape[-2], x.shape[-1]) for x in breakdown_feats]

        # Predict transition regions with dynamic kernels and features
        transition_preds = self.forward_trans_pred(breakdown_feats, inc_kernels, masks)
        inc_bin_map = [transition_preds.sigmoid() > 0.5]

        return [transition_preds], inc_bin_map
    
        # Select points based on gt_trans during training
        
        # and transition_preds during testing

        import pdb; pdb.set_trace()
        
        # Generate position emebddings and add to features to have point features to refine
        point_feats = []
        point_pe = []

        for i, feat in enumerate(breakdown_feats):
            pe_emb = self.pe_layers[i](bs, n_f, feat.shape[-2], feat.shape[-1], feat.device)
            point_pe += [pe_emb.flatten(2)]
            point_feats += [feat.squeeze(2).permute(0, 2, 1, 3, 4).flatten(2)]

        point_feats = torch.cat(point_feats, dim=2)
        point_pe = torch.cat(point_pe, dim=2)
        
        inc_feats = torch.gather(point_feats.unsqueeze(1) \
                                            .expand(-1, n_ints, -1, -1), dim=3, \
                                 index=inc_indices.unsqueeze(2) \
                                                  .expand(-1, -1, point_feats.shape[1], -1))
        inc_pe = torch.gather(point_pe.unsqueeze(1) \
                                      .expand(-1, n_ints, -1, -1), dim=3, \
                              index=inc_indices.unsqueeze(2)\
                                               .expand(-1, -1, point_pe.shape[1], -1))

        return [transition_preds], inc_bin_map, inc_feats, inc_pe, inc_indices

    def pixel_decoder_forward(self, features, weights, biases):
        '''
        features: (b * n_ins, n_points, c)
        weights: [(b * n_ins, c_i, c_j)]
        biases: [(b * n_ins, n_points, c_j)]
        '''
        assert features.dim() == 3
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = torch.bmm(x, w) + b
            if i < n_layers - 1:
                x = F.relu(x)
        return x
    
    def refine_pixel(self, inc_feats, inc_pe, decoder_kernels):
        '''
        inc_feats: (b, n_instances, c, n_points)
        inc_pe: (b, n_instances, c, n_points)

        Returns:
        refined_points: (b, n_instances, 1, n_points)
        '''
        b, n_ints, _, n_points = inc_feats.shape

        # Reshape to (n_instances * n_points, b, c)
        inc_feats = inc_feats.permute(1, 3, 0, 2).reshape(-1, inc_feats.shape[0], inc_feats.shape[2])
        inc_pe = inc_pe.permute(1, 3, 0, 2).reshape(-1, inc_pe.shape[0], inc_pe.shape[2])

        inc_feats = self.pixel_encoder(inc_feats, inc_pe)
        
        inc_feats = inc_feats.reshape(n_ints, n_points, b, -1) \
                             .permute(2, 0, 1, 3) \
                             .reshape(b * n_ints, n_points, -1)

        weights, biases = self.parse_pixdec_dynamic_params(decoder_kernels, [32 * 8, 8 * 4, 4 * 1], [8, 4, 1])

        inc_logit = self.pixel_decoder_forward(inc_feats, weights, biases)
        inc_logit = inc_logit.reshape(b, n_ints, n_points)
        
        return inc_logit
    
    def _refine_masks(self, masks, inc_ids, id_l_bound, id_u_bound, refined_logit_inc):
        
        b, n_instances, n_f, h, w = masks.shape

        # Change masks shape to (b, n_instances, n_f * h * w) for refinement
        masks = masks.flatten(2)

        b_ids = torch.arange(b)[:, None, None].expand(-1, n_instances, inc_ids.shape[-1]).to(masks.device)
        ins_ids = torch.arange(n_instances)[None, :, None].expand(b, -1, inc_ids.shape[-1]).to(masks.device)
        select_id_mask = (inc_ids < id_u_bound) & (inc_ids >= id_l_bound)
        
        spa_ids = inc_ids[select_id_mask]
        b_ids = b_ids[select_id_mask]
        ins_ids = ins_ids[select_id_mask]

        if len(spa_ids) > 0:
            masks[b_ids, ins_ids, spa_ids - id_l_bound] = refined_logit_inc[select_id_mask].sigmoid()

        return masks.reshape(b, n_instances, n_f, h, w)

    def refine_masks(self, masks, refined_logit_inc, inc_ids):
        
        _, n_f, _, h, w = masks.shape

        # Update masks
        masks = masks.float().clone()
        
        # Change masks shape to (b, n_instances, n_f, h, w) for refinement
        masks = masks.permute(0, 2, 1, 3, 4)

        # Refine masks OS8
        lower_bound = 0
        upper_bound = 0
        
        for u in [2, 4, 1]:
            lower_bound += upper_bound
            upper_bound += n_f * h * w
            masks = self._refine_masks(masks, inc_ids, lower_bound, upper_bound, refined_logit_inc)
            h *= u
            w *= u
            if u > 1:
                masks = self.upsample_bin_map(masks, u)

        return masks.permute(0, 2, 1, 3, 4)

    def forward(self, batch):
        '''
        x: b, n_f, 3, h, w, image tensors
        masks: b, n_frames, n_instances, h//8, w//8, coarse masks
        alphas: b, n_frames, n_instances, h, w, alpha matte
        trans_gt: b, n_frames, n_instances, h, w, incoherence mask ground truth
        '''
        x = batch['image']
        masks = batch['mask']
        alphas = batch.get('alpha', None)
        trans_gt = batch.get('transition', None)

        # Forward image to get features
        b, n_f, _, h, w = x.shape
        n_instances = masks.shape[2]
        
        # Reshape x and masks
        x = x.reshape(b * n_f, 3, h, w)
        # masks = masks.reshape(b,n_f, n_instances, h//8, w//8)
        feats = self.backbone(x) # os1 to os32

        # TODO: Running FPN to have attention features (1/16, 1/8, 1/4) and breakdown features (1/8, 1/4, 1)
        attention_feats, breakdown_feats = self.fpn(feats)

        # attention_feats = self.extract_features(feats, self.atten_features, self.conv_atten)
        # breakdown_feats = self.extract_features(feats, self.breakdown_features, self.conv_breakdown)

        # Compute dynamic kernels weights with attention module
        incoherence_kernels, decoder_kernels = self.dynamic_kernels_generator(attention_feats, masks)
        # incoherence_kernels: b, n_instances, kernel_size
        # decoder_kernels: b, n_instances, kernel_size

        output = {}

        # Detect incoherent regions
        trans_preds, inc_bin_map = self.inc_prediction(incoherence_kernels, masks, breakdown_feats)
        output['trans_preds'] = trans_preds
        output['inc_bin_maps'] = inc_bin_map

        # high-res features for instances
        os_4up = F.interpolate(feats['os4'], size=(h, w), mode='bilinear', align_corners=True)
        hr_feat = self.hr_conv(torch.cat([os_4up, feats['os1']], dim=1))
        hr_feat = hr_feat.unsqueeze(1).expand(-1, n_instances, -1, -1, -1).flatten(3).reshape(b * n_f * n_instances, 64, -1).permute(0, 2, 1)
        decoder_kernels = decoder_kernels.unsqueeze(1).expand(-1, n_f, -1, -1).reshape(b * n_f * n_instances, -1).unsqueeze(-1)
        atten_map = torch.bmm(hr_feat, decoder_kernels).reshape(b, n_f, n_instances, h, w).unsqueeze(3).sigmoid()
        hr_feat = hr_feat.reshape(b, n_f, n_instances, h, w, -1).permute(0, 1, 2, 5, 3, 4)
        hr_feat = hr_feat * (1 + atten_map)

        # Finding poinst to refine
        interested_pt = None
        if self.training and trans_gt is not None:
            interested_pt = trans_gt
        else:
            interested_pt = inc_bin_map[0].float()
            interested_pt = self.upsample_bin_map(interested_pt, 8)
        l_masks = self.upsample_bin_map(masks, 8)
        hr_feat = torch.cat([hr_feat, l_masks.unsqueeze(3)], dim=3)
        hr_feat = hr_feat.reshape(b * n_f * n_instances, -1, h, w)
        indices = interested_pt.reshape(b * n_f * n_instances, -1, h, w)
        
        indices = torch.where(indices.squeeze(1) > 0)
        hr_feat = hr_feat.permute(0,2,3,1)
        hr_feat = hr_feat[indices]
        indices = torch.stack(indices, dim=1)
        if len(indices) == 0:
            output['refined_masks'] = l_masks
            if self.training:
                return output, None
            return output
        x = spconv.SparseConvTensor(hr_feat, indices.int(), (h, w), b * n_f * n_instances)
        x = self.pixel_decoder(x)
        x = x.dense()
        x = x.reshape(b, n_f, n_instances, h, w)
        x = x * interested_pt + (1.0 - interested_pt) * l_masks
        
        output['refined_masks'] = x
        
        if self.training:
            return output, self.compute_loss(output, alphas, trans_gt)
        return output
        

    def _compute_trans_loss(self, trans_preds, trans_gt):
        trans_loss = 0
        for trans_pred in trans_preds:
            b, n_f, n_i, h, w = trans_pred.shape
            trans_pred = trans_pred.reshape(b * n_f * n_i, 1, h, w)
            gt = self.downsample_bin_map(trans_gt, trans_gt.shape[-1] // w)
            pos_weight = torch.Tensor([3.0]).to(trans_pred.device).unsqueeze(-1).unsqueeze(-1)
            gt = gt.reshape(b * n_f * n_i, 1, h, w)
            trans_loss = trans_loss + binary_focal_loss_with_logits(trans_pred, gt, reduction='mean', pos_weight=pos_weight)

        return trans_loss

    def _query_gt(self, alphas, refined_logit_inc, inc_ids, id_l_bound, id_u_bound):
        b, n_instances, n_f, h, w = alphas.shape

        # Change masks shape to (b, n_instances, n_f * h * w) for refinement
        alphas = alphas.flatten(2)

        b_ids = torch.arange(b)[:, None, None].expand(-1, n_instances, inc_ids.shape[-1]).to(alphas.device)
        ins_ids = torch.arange(n_instances)[None, :, None].expand(b, -1, inc_ids.shape[-1]).to(alphas.device)
        select_id_mask = (inc_ids < id_u_bound) & (inc_ids >= id_l_bound)
        
        spa_ids = inc_ids[select_id_mask]
        b_ids = b_ids[select_id_mask]
        ins_ids = ins_ids[select_id_mask]

        if len(spa_ids) > 0:
            return alphas[b_ids, ins_ids, spa_ids - id_l_bound], refined_logit_inc[select_id_mask].sigmoid()

        return None, None
    
    def _compute_refinement_loss(self, refined_logit_inc, alphas, inc_ids):
        b, n_f, n_i, h, w = alphas.shape
        alphas = alphas.permute(0, 2, 1, 3, 4)
        alphas = alphas.reshape(b * n_i * n_f, 1, h, w)

        # Compute refinement loss (L1 loss): refined_logit_inc, alphas, inc_ids
        lower_bound = 0
        upper_bound = 0
        ref_loss = 0
        for d in [0.125, 0.25, 1]:
            lower_bound += upper_bound
            upper_bound += n_f * int(h * d) * int(w * d)

            if d == 1:
                d_alpha = alphas
            else:
                d_alpha = F.interpolate(alphas, size=(int(h * d), int(w * d)), mode='bilinear', align_corners=True)
            d_alpha = d_alpha.reshape(b, n_i, n_f, int(h * d), int(w * d))
            gt, pred = self._query_gt(d_alpha, refined_logit_inc, inc_ids, lower_bound, upper_bound)
            if gt is not None:
                ref_loss = ref_loss + F.l1_loss(pred.flatten(), gt.flatten(), reduction='sum')
        
        return ref_loss / inc_ids.numel()
    
    
    def compute_loss(self, output_dict, alphas, trans_gt):
        losses = {}
        
        total_loss = 0
        # Compute transition loss (binary cross entropy): trans_preds, trans_gt
        trans_loss = self._compute_trans_loss(output_dict['trans_preds'], trans_gt)
        losses['loss_trans'] = trans_loss
        total_loss = total_loss + trans_loss
        
        # Alpha loss: L1 loss
        if self.loss_alpha_w > 0:
            ref_loss = self.alpha_loss(output_dict['refined_masks'], alphas)
            ref_loss = (ref_loss * trans_gt).sum() / trans_gt.sum()
            losses['loss_ref_alpha'] = ref_loss
            total_loss = total_loss + ref_loss * self.loss_alpha_w

        # Alpha loss: gradient loss
        if self.loss_alpha_grad_w > 0:
            h, w = alphas.shape[-2:]
            alpha_pred = output_dict['refined_masks'].reshape(-1, 1, h, w)
            alpha_gt = alphas.reshape(-1, 1, h, w)
            mask = trans_gt.reshape(-1, 1, h, w)
            grad_loss = self.gradient_loss(alpha_pred, alpha_gt, mask)
            losses['loss_ref_grad'] = grad_loss
            total_loss = total_loss + grad_loss * self.loss_alpha_grad_w
        
        # Alpha loss: laplacian loss
        if self.loss_alpha_lap_w > 0:
            h, w = alphas.shape[-2:]
            pred_alpha = output_dict['refined_masks'].reshape(-1, 1, h, w)
            gt_alpha = alphas.reshape(-1, 1, h, w)
            lap_loss = self.lap_loss(pred_alpha, gt_alpha)
            losses['loss_ref_lap'] = lap_loss
            total_loss = total_loss + lap_loss * self.loss_alpha_lap_w

        losses['total'] = total_loss
        return losses
        