from functools import partial
import random

import torch
from torch import nn
from torch.nn import functional as F

from .resnet import BasicBlock
from .resnet_inst_matt_spconv import ResShortCut_InstMattSpconv_Dec
from ..loss import loss_dtSSD
from ..module import SpectralNorm, conv1x1, conv3x3, ConvGRU
from ...utils.utils import compute_unknown, gaussian_smoothing

class ResShortCut_InstMattSpconv_BiTempSpar_Dec(ResShortCut_InstMattSpconv_Dec):
    def __init__(self, temp_method='bi', **kwargs):
        super().__init__(use_temp=True, **kwargs)
        
        self.temp_method = temp_method.split("_")[0]
        self.use_fusion = 'fusion' in temp_method
        self.use_temp = temp_method != 'none'

        self.os8_temp_module = ConvGRU(128, dilation=1, padding=1)

        # Module to compute the difference between two inputs
        self.diff_module = nn.Sequential(
            SpectralNorm(conv1x1(128, 64)),
            self._norm_layer(64),
            nn.ReLU(inplace=True),
            SpectralNorm(conv3x3(64, 32)),
            self._norm_layer(32),
            nn.ReLU(inplace=True),
            conv3x3(32, 1)
        )

    def bidirectional_fusion(self, feat, preds):
        '''
        Fuse forward and backward features
        '''
        n_f = feat.shape[1]
        forward_diffs = []
        backward_diffs = []
        forward_preds = [preds[:, 0]]
        backward_preds = [preds[:, n_f-1]]
        fuse_preds = []

        # forward
        for i in range(1, n_f):
            diff = self.diff_module(torch.cat([feat[:, i-1], feat[:, i]], dim=1))
            diff = F.interpolate(diff, scale_factor=8.0, mode='bilinear', align_corners=False)
            forward_diffs.append(diff)
            pred = forward_preds[-1] * (1 - diff.sigmoid()) + preds[:, i] * diff.sigmoid()
            forward_preds.append(pred)
        
        forward_diffs = [torch.zeros_like(forward_diffs[0])] + forward_diffs
        forward_diffs = torch.stack(forward_diffs, dim=1)

        # backward
        for i in range(n_f-1, 0, -1):
            diff = self.diff_module(torch.cat([feat[:, i], feat[:, i-1]], dim=1))
            diff = F.interpolate(diff, scale_factor=8.0, mode='bilinear', align_corners=False)
            backward_diffs.append(diff)
            pred = backward_preds[-1] * (1 - diff.sigmoid()) + preds[:, i-1] * diff.sigmoid()
            backward_preds.append(pred)
        
        backward_preds = backward_preds[::-1]
        backward_diffs = backward_diffs[::-1]
        backward_diffs = backward_diffs + [torch.zeros_like(backward_diffs[-1])]
        backward_diffs = torch.stack(backward_diffs, dim=1)

        # Fuse forward and backward
        for i in range(n_f):
            if i == 0:
                fuse_preds.append(forward_preds[i])
            elif i == n_f - 1:
                fuse_preds.append(backward_preds[i])
            else:
                fuse_preds.append((forward_preds[i] + backward_preds[i]) / 2)
        fuse_preds = torch.stack(fuse_preds, dim=1)
        return forward_diffs, backward_diffs, fuse_preds

    def forward(self, x, mid_fea, b, n_f, n_i, masks, iter, gt_alphas, mem_feat=None, spar_gt=None, **kwargs):
        
        '''
        masks: [b * n_f * n_i, 1, H, W]
        '''
        x, masks, valid_masks, gt_masks, fea1, fea2, fea3, image, h, w = self.os32_to_os8(x, mid_fea, b, n_f, n_i, masks, gt_alphas)
        
        # Predict OS8
        temp_propagate_fn = partial(self.os8_temp_module.propagate_features, n_f=n_f, prev_h_state=mem_feat, temp_method=self.temp_method)
        x_os8, x, hidden_state, queries, loss_max_atten, hidden_state = self.refine_OS8(x, masks, use_mask_atten=False, gt_mask=gt_masks, 
                                                                            aggregate_mem_fn=temp_propagate_fn)
        
        mem_feat = hidden_state

        # Predict temporal sparsity here, forward and backward
        feat_os8 = x.view(b, n_f, *x.shape[1:]).detach()
            
        # Upsample - normalize OS8 pred
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0
        
        if self.training:
            x_os8 = x_os8 * valid_masks
        else:
            x_os8 = x_os8[:, :n_i]

        # Warm-up - Using gt_alphas instead of x_os8 for later steps
        guided_mask_os8 = x_os8
        is_use_alphas_gt = False
        if self.training and ((iter < self.warmup_detail_iter or x_os8.sum() == 0 or (iter < self.warmup_detail_iter * 3 and random.random() < 0.5))):
            guided_mask_os8 = gt_alphas.clone()
            is_use_alphas_gt = True
        
        # Compute unknown regions
        upper_thres = 0.95
        if not self.training:
            x_os8[x_os8 >= upper_thres] = 1.0

        unknown_os8 = compute_unknown(guided_mask_os8, k_size=30)

        # Ignore out of the box region
        if not self.training:
            h, w = image.shape[-2:]
            thresh = 0.1
            padding = 30
            smooth_os8 = gaussian_smoothing(x_os8, sigma=3)
            for i in range(smooth_os8.shape[0]):
                for j in range(n_i):
                    coarse_inst = smooth_os8[i, j] > thresh
                    ys, xs = torch.nonzero(coarse_inst, as_tuple=True)
                    if len(ys) == 0:
                        continue
                    y_min, y_max = ys.min(), ys.max()
                    x_min, x_max = xs.min(), xs.max()
                    y_min = max(0, y_min - padding)
                    y_max = min(y_max + padding, h)
                    x_min = max(0, x_min - padding)
                    x_max = min(x_max + padding, w)
                    target_mask = torch.zeros_like(coarse_inst)
                    target_mask[y_min: y_max, x_min: x_max] = 1
                    unknown_os8[i, j] = unknown_os8[i, j] * target_mask
                    x_os8[i, j] = x_os8[i, j] * target_mask
                    
        x_os4, x_os1 = self.process_os4_os1(x, b, n_f, fea1, fea2, fea3, image, x_os8, queries, guided_mask_os8, unknown_os8)

        ret = {}
        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8

        # Fusion
        alpha_pred, weight_os4, weight_os1 = self.fuse(ret, unknown_os8)
        ret['refined_masks'] = alpha_pred
        ret['detail_mask'] = unknown_os8
        
        if self.use_temp:
            ret['mem_feat'] = mem_feat
        
        if is_use_alphas_gt:
            weight_os4 = compute_unknown(gt_alphas, k_size=30, is_train=self.training) * unknown_os8
            weight_os1 = compute_unknown(gt_alphas, k_size=15, is_train=self.training) * unknown_os8
        ret['weight_os4'] = weight_os4
        ret['weight_os1'] = weight_os1 

        # Fuse temporal sparsity
        temp_alpha = alpha_pred.view(b, n_f, *alpha_pred.shape[1:])
        diff_forward, diff_backward, temp_fused_alpha = self.bidirectional_fusion(feat_os8, temp_alpha)
        
        if (not self.training and self.use_fusion) or self.training:
            ret['temp_alpha'] = temp_fused_alpha
            ret['diff_forward'] = diff_forward.sigmoid()
            ret['diff_backward'] = diff_backward.sigmoid()

        # Adding some losses to the results
        if self.training:
            ret['loss_max_atten'] = loss_max_atten

            # Compute loss for temporal sparsity
            ret.update(self.loss_temporal_sparsity(diff_forward, diff_backward, spar_gt))

        return ret
    
    def loss_temporal_sparsity(self, diff_forward, diff_backward, spar_gt):
        '''
        Compute loss for temporal sparsity
        '''
        loss = {}

        # BCE loss
        spar_gt = spar_gt.view(diff_forward.shape[0], -1, *spar_gt.shape[1:])
        bce_forward_loss = F.binary_cross_entropy_with_logits(diff_forward[:, 1:, 0], spar_gt[:, 1:, 0], reduction='mean')
        bce_backward_loss = F.binary_cross_entropy_with_logits(diff_backward[:, :-1, 0], spar_gt[:, 1:, 0], reduction='mean')
        loss['loss_temp_bce'] = bce_forward_loss + bce_backward_loss

        # Fusion loss
        # import pdb; pdb.set_trace()
        dtSSD_forward = loss_dtSSD(diff_forward[:, 1:].sigmoid(), spar_gt[:, 1:, 0:1], torch.ones_like(spar_gt[:, 1:, 0:1]))
        dtSSD_backward = loss_dtSSD(diff_backward[:, :-1].sigmoid(), spar_gt[:, 1:, 0:1], torch.ones_like(spar_gt[:, 1:, 0:1]))
        loss['loss_temp_dtssd'] = dtSSD_forward + dtSSD_backward

        loss['loss_temp'] = (loss['loss_temp_bce'] + dtSSD_forward + dtSSD_backward) * 0.25

        return loss


def res_shortcut_inst_matt_spconv_temp_22(**kwargs):
    return ResShortCut_InstMattSpconv_BiTempSpar_Dec(block=BasicBlock, layers=[2, 3, 3, 2], **kwargs)