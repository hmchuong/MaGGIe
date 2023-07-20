import math
import random
import logging
# from pudb.remote import set_trace

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from kornia.losses import binary_focal_loss_with_logits
import spconv.pytorch as spconv
from vm2m.network.loss import GradientLoss, LapLoss, RMSELoss, loss_dtSSD, loss_comp
from vm2m.network.module.aspp import ASPP

from vm2m.network.module.dcn_idk import DCInstDynKernelGenerator
from vm2m.network.module.temporal_attention import KernelTemporalAttention
from vm2m.utils import resizeAnyShape

from .mgm import get_unknown_tensor_from_pred

class VM2M0711(nn.Module):
    def __init__(self, backbone, decoder, cfg):
        super().__init__()
        
        # Backbone module
        self.backbone = backbone
        self.cfg = cfg

        self.aspp = ASPP(in_channel=cfg.aspp.in_channels, out_channel=cfg.aspp.out_channels)
        
        # Dynamic kernel generator
        self.ik_generator = DCInstDynKernelGenerator(cfg.aspp.out_channels, 
                                                     cfg.dynamic_kernel.hidden_dim, 
                                                     cfg.dynamic_kernel.out_incoherence,
                                                     cfg.dynamic_kernel.out_pixeldecoder)
        
        # 512 OS32 -> 32 OS8 ->  32 -> 4  + 4 -> 1
        self.inc_conv = nn.Sequential(
            nn.PixelShuffle(4),
            nn.Conv2d(cfg.aspp.out_channels//16, cfg.aspp.out_channels//16, 3, padding=1),
            nn.BatchNorm2d(cfg.aspp.out_channels//16),
            nn.ReLU(),
        )

        self.inc_smooth = nn.Conv2d(2, 1, 5, padding=2)

        self.inc_attention = KernelTemporalAttention(cfg.dynamic_kernel.out_incoherence)
        self.dec_attention = KernelTemporalAttention(cfg.dynamic_kernel.out_pixeldecoder)
        self.decoder = decoder

        # Losses
        self.gradient_loss = GradientLoss()
        self.lap_loss = LapLoss()
        self.alpha_loss = nn.L1Loss(reduction='none') if cfg.loss_alpha_type == 'l1' else RMSELoss()
        self.loss_alpha_w = cfg.loss_alpha_w
        self.loss_alpha_grad_w = cfg.loss_alpha_grad_w
        self.loss_alpha_lap_w = cfg.loss_alpha_lap_w
        self.loss_dtSSD_w = cfg.loss_dtSSD_w
        self.loss_comp_w = cfg.loss_comp_w
        
        need_init_weights = [self.aspp, self.decoder, self.ik_generator]

        # Init weights
        for module in need_init_weights:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # def predict_inc(self, embedding, inc_kernels):
    #     '''
    #     embedding: b x n_f, c, h/32, w/32
    #     inc_kernels: b, n_i, d

    #     Return:
    #     x: b x n_f, n_i, h/8, w/8
    #     '''
    #     x = self.inc_conv(embedding) # x: b x n_f, c/16, h/8, w/8
        
    #     c, h, w = x.shape[-3:]
    #     b, n_i = inc_kernels.shape[:2]
    #     n_f = x.shape[0] // b
        
    #     x = x.reshape(b * n_f, -1, c, h, w)
    #     x = x.repeat(1, n_i, 1, 1, 1) # b*n_f, n_i, c, h, w
    #     x = x.permute(0, 1, 3, 4, 2)
    #     x = x.reshape(b * n_f * n_i, h * w, c) # b*n_f*n_i, h*w, c
    #     inc_kernels = inc_kernels.unsqueeze(1) # b, 1, n_i, d
    #     inc_kernels = inc_kernels.repeat(1, n_f, 1, 1) # b, n_f, n_i, d
    #     inc_kernels = inc_kernels.reshape(b * n_f * n_i, -1, 1) # b*n_f*n_i, d

    #     # Split parameters
    #     param_splits = torch.split_with_sizes(inc_kernels, split_sizes=[32 * 4, 4 * 1, 4, 1], dim=1)
    #     inc_weights = param_splits[:2]
    #     inc_biases = param_splits[2:]

    #     for i in range(2):
    #         inc_w = inc_weights[i].reshape(x.shape[0], x.shape[-1], -1)
    #         inc_b = inc_biases[i].reshape(x.shape[0], 1, -1)
    #         x = torch.bmm(x, inc_w) + inc_b
    #         x = F.relu(x)
        
    #     # x: b*n_f*n_i, h*w, 1
    #     x = x.reshape(b * n_f, n_i, h, w)
    #     return x
    
    def predict_inc(self, embedding, inc_kernels, coarse_masks):
        '''
        embedding: b x n_f, c, h/32, w/32
        inc_kernels: b, n_f n_i, d

        Return:
        x: b x n_f, n_i, h/8, w/8
        '''
        x = self.inc_conv(embedding) # x: b x n_f, c/16, h/8, w/8

        c, h, w = x.shape[-3:]
        b, n_f, n_i = inc_kernels.shape[:3]

        inc_kernels = inc_kernels.reshape(b * n_f * n_i, -1) # b*n_f*n_i, d
        param_splits = torch.split_with_sizes(inc_kernels, split_sizes=[32 * 4, 4 * 1, 4, 1], dim=1)
        inc_weights = param_splits[:2]
        inc_biases = param_splits[2:]

        x = x.reshape(1, -1, h, w)
        n_layers = len(inc_weights)
        
        for i in range(n_layers):
            input_shape = x.shape[1] // (b * n_f)
            weight = inc_weights[i].reshape(b * n_f * n_i, -1, input_shape, 1, 1)
            weight = weight.reshape(-1, input_shape, 1, 1)
            bias = inc_biases[i].flatten()
            x = F.conv2d(x, weight=weight, bias=bias, stride=1, padding=0, groups=b * n_f)
            if i < n_layers - 1:
                x = F.relu(x)

        # Smooth with a large conv kernel
        x = x.reshape(b * n_f * n_i, 1, h, w)
        masks = coarse_masks.reshape(b * n_f * n_i, 1, h, w)
        x = torch.cat([x.sigmoid(), masks], dim=1)
        x = self.inc_smooth(x)

        x = x.reshape(b * n_f, n_i, h, w)
        return x

    def fushion(self, pred):
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        ### Progressive Refinement Module in MGMatting Paper
        alpha_pred = alpha_pred_os8.clone().detach()
        weight_os4 = get_unknown_tensor_from_pred(alpha_pred, rand_width=30, train_mode=self.training)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4> 0]
        weight_os1 = get_unknown_tensor_from_pred(alpha_pred, rand_width=15, train_mode=self.training)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1 > 0]

        return alpha_pred, weight_os4, weight_os1

    def gen_unknown_region(self, masks):
        '''
        masks: b * n_f * n_i, h, w
        '''
        
        mask_cpu = masks.cpu().numpy()
        unknown_masks = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        for i in range(mask_cpu.shape[0]):
            dilated = cv2.dilate(mask_cpu[i, 0], kernel)
            eroded = cv2.erode(mask_cpu[i, 0], kernel)
            unknown_masks.append(dilated - eroded)
        unknown_masks = np.stack(unknown_masks, axis=0)
        unknown_masks = torch.from_numpy(unknown_masks).to(masks.device)
        unknown_masks = (unknown_masks > 0).float()
        return unknown_masks

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
        fg = batch.get('fg', None)
        bg = batch.get('bg', None)

        # Forward image to get features
        b, n_f, _, h, w = x.shape
        n_i = masks.shape[2]
        
        # Reshape x and masks
        x = x.reshape(b * n_f, 3, h, w)

        if fg is not None:
            fg = fg.view(-1, 3, h, w)
        if bg is not None:
            bg = bg.view(-1, 3, h, w)
        
        embedding, mid_fea = self.backbone(x)
        embedding = self.aspp(embedding)

        # Generate dynamic kernels
        inc_kernels, dec_kernels = self.ik_generator(embedding, masks)

        # Channel-wise Temporal attention between same-int frames
        inc_kernels = self.inc_attention(inc_kernels)
        dec_kernels = self.dec_attention(dec_kernels)

        

        # Predict inc_mask
        inc_pred = self.predict_inc(embedding, inc_kernels, masks) # b*n_f, n_i, h/8, w/8
        
        # Use gt inc_mask if training
        if self.training:
            inc_mask = resizeAnyShape(trans_gt, scale_factor=1.0/8.0, use_max_pool=True)
            inc_mask = inc_mask.reshape(*inc_pred.shape)
        else:
            inc_mask = (inc_pred.sigmoid() > 0.5).float()

            # Get input mask dilation + erosion
            unk_mask = self.gen_unknown_region(masks.reshape(b * n_f * n_i, 1, h//8, w//8))
            unk_mask = unk_mask.reshape(*inc_mask.shape)
            inc_mask = ((inc_mask + unk_mask) > 0.5).float()

        if inc_mask.sum() > 0:
            dec_kernels = dec_kernels.reshape(b * n_f, n_i, -1)
            preds = self.decoder(embedding, mid_fea, inc_mask, masks.reshape(*inc_pred.shape), b, n_f, n_i, dec_kernels)
        else:
            masks = masks.reshape(b * n_f, n_i, h//8, w//8)
            masks = resizeAnyShape(masks, scale_factor=8.0)
            preds = {}
            preds['alpha_os1'] = masks
            preds['alpha_os4'] = masks
            preds['alpha_os8'] = masks
        output = {}
        output['alpha_os1'] = preds['alpha_os1'].view(b, n_f, n_i, h, w)
        output['alpha_os4'] = preds['alpha_os4'].view(b, n_f, n_i, h, w)
        output['alpha_os8'] = preds['alpha_os8'].view(b, n_f, n_i, h, w)

        alpha_pred, weight_os4, weight_os1 = self.fushion(preds)
        alpha_pred = alpha_pred.view(b, n_f, n_i, h, w)
        output['refined_masks'] = alpha_pred

        output['trans_preds'] = [inc_pred.view(b, n_f, n_i, h//8, w//8)]
        output['inc_bin_maps'] = [inc_mask.view(b, n_f, n_i, h//8, w//8)]

        if self.training:
            iter = batch['iter']
            loss_dict = self.compute_loss(inc_pred, preds, weight_os4, weight_os1, alphas, inc_mask, trans_gt, fg, bg, iter)
            return output, loss_dict
        
        return output
    
    def _compute_trans_loss(self, trans_preds, trans_gt):
        pos_weight = torch.Tensor([3.0]).to(trans_preds.device).unsqueeze(-1).unsqueeze(-1)
        trans_loss = binary_focal_loss_with_logits(trans_preds, trans_gt, reduction='mean', pos_weight=pos_weight)

        return trans_loss

    @staticmethod
    def regression_loss(logit, target, loss_type='l1', weight=None):
        """
        Alpha reconstruction loss
        :param logit:
        :param target:
        :param loss_type: "l1" or "l2"
        :param weight: tensor with shape [N,1,H,W] weights for each pixel
        :return:
        """
        if weight is None:
            if loss_type == 'l1':
                return F.l1_loss(logit, target)
            elif loss_type == 'l2':
                return F.mse_loss(logit, target)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
        else:
            if loss_type == 'l1':
                return F.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            elif loss_type == 'l2':
                return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
       
    def compute_loss(self, inc_pred, pred, weight_os4, weight_os1, alphas_ori, inc_gt, trans_gt_ori, fg, bg, iter):
        '''
        pred: dict of output from forward
        batch: dict of input batch
        '''
        h, w = alphas_ori.shape[-2:]
        alphas = alphas_ori.view(-1, 1, h, w)
        trans_gt = trans_gt_ori.view(-1, 1, h, w)

        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        if iter < self.cfg.mgm.warmup_iter or (iter < self.cfg.mgm.warmup_iter * 3 and random.randint(0,1) == 0):
            weight_os1 = trans_gt
            weight_os4 = trans_gt
            # logging.debug('Using ground truth mask')
        else:
            alpha_pred_os4[weight_os4==0] = alpha_pred_os8[weight_os4==0]
            alpha_pred_os1[weight_os1==0] = alpha_pred_os4[weight_os1==0]
            # logging.debug('Using prediction mask')

        loss_dict = {}
        weight_os8 = torch.ones_like(weight_os1)

        
        total_loss = 0

        trans_loss = self._compute_trans_loss(inc_pred, inc_gt)
        loss_dict['loss_trans'] = trans_loss
        total_loss = total_loss + trans_loss

        # Reg loss
        if self.loss_alpha_w > 0:
            ref_alpha_os1 = self.regression_loss(alpha_pred_os1, alphas, loss_type=self.cfg.loss_alpha_type, weight=weight_os1)
            ref_alpha_os4 = self.regression_loss(alpha_pred_os4, alphas, loss_type=self.cfg.loss_alpha_type, weight=weight_os4)
            ref_alpha_os8 = self.regression_loss(alpha_pred_os8, alphas, loss_type=self.cfg.loss_alpha_type, weight=weight_os8)
            ref_alpha_loss = (ref_alpha_os1 * 2 + ref_alpha_os4 * 1 + ref_alpha_os8 * 1) / 5.0
            loss_dict['loss_rec_os1'] = ref_alpha_os1
            loss_dict['loss_rec_os4'] = ref_alpha_os4
            loss_dict['loss_rec_os8'] = ref_alpha_os8
            loss_dict['loss_rec'] = ref_alpha_loss
            total_loss += ref_alpha_loss * self.loss_alpha_w
        
        # Grad loss
        if self.loss_alpha_grad_w > 0:
            grad_loss_os1 = self.gradient_loss(alpha_pred_os1, alphas, weight_os1)
            grad_loss_os4 = self.gradient_loss(alpha_pred_os4, alphas, weight_os4)
            grad_loss_os8 = self.gradient_loss(alpha_pred_os8, alphas, weight_os8)
            grad_loss = (grad_loss_os1 * 2 + grad_loss_os4 * 1 + grad_loss_os8 * 1) / 5.0
            loss_dict['loss_grad_os1'] = grad_loss_os1
            loss_dict['loss_grad_os4'] = grad_loss_os4
            loss_dict['loss_grad_os8'] = grad_loss_os8
            loss_dict['loss_grad'] = grad_loss
            total_loss += grad_loss * self.loss_alpha_grad_w
        
        # Comp loss
        if self.loss_comp_w > 0 and fg is not None and bg is not None:
            comp_loss_os1 = loss_comp(alpha_pred_os1, alphas, fg, bg, weight_os1)
            comp_loss_os4 = loss_comp(alpha_pred_os4, alphas, fg, bg, weight_os4)
            comp_loss_os8 = loss_comp(alpha_pred_os8, alphas, fg, bg, weight_os8)
            comp_loss = (comp_loss_os1 * 2 + comp_loss_os4 * 1 + comp_loss_os8 * 1) / 5.0
            loss_dict['loss_comp_os1'] = comp_loss_os1
            loss_dict['loss_comp_os4'] = comp_loss_os4
            loss_dict['loss_comp_os8'] = comp_loss_os8
            loss_dict['loss_comp'] = comp_loss
            total_loss += comp_loss * self.loss_comp_w

        # Lap loss
        if self.loss_alpha_lap_w > 0:
            lap_loss_os1 = self.lap_loss(alpha_pred_os1, alphas, weight_os1)
            lap_loss_os4 = self.lap_loss(alpha_pred_os4, alphas, weight_os4)
            lap_loss_os8 = self.lap_loss(alpha_pred_os8, alphas, weight_os8)
            lap_loss = (lap_loss_os1 * 2 + lap_loss_os4 * 1 + lap_loss_os8 * 1) / 5.0
            loss_dict['loss_lap_os1'] = lap_loss_os1
            loss_dict['loss_lap_os4'] = lap_loss_os4
            loss_dict['loss_lap_os8'] = lap_loss_os8
            loss_dict['loss_lap'] = lap_loss
            total_loss += lap_loss * self.loss_alpha_lap_w
        
        # Compute temporal loss
        if self.loss_dtSSD_w > 0:
            alpha_pred_os8 = alpha_pred_os8.reshape(*alphas_ori.shape)
            alpha_pred_os4 = alpha_pred_os4.reshape(*alphas_ori.shape)
            alpha_pred_os1 = alpha_pred_os1.reshape(*alphas_ori.shape)
            # import pdb; pdb.set_trace()
            dtSSD_loss_os1 = loss_dtSSD(alpha_pred_os1, alphas_ori, trans_gt_ori)
            dtSSD_loss_os4 = loss_dtSSD(alpha_pred_os4, alphas_ori, trans_gt_ori)
            dtSSD_loss_os8 = loss_dtSSD(alpha_pred_os8, alphas_ori, trans_gt_ori)
            dtSSD_loss = (dtSSD_loss_os1 * 2 + dtSSD_loss_os4 * 1 + dtSSD_loss_os8 * 1) / 5.0
            loss_dict['loss_dtSSD_os1'] = dtSSD_loss_os1
            loss_dict['loss_dtSSD_os4'] = dtSSD_loss_os4
            loss_dict['loss_dtSSD_os8'] = dtSSD_loss_os8
            loss_dict['loss_dtSSD'] = dtSSD_loss
            total_loss = total_loss + dtSSD_loss * self.loss_dtSSD_w

        loss_dict['total'] = total_loss
        return loss_dict