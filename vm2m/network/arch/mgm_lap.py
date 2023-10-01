from functools import partial
import logging
import numpy as np
import cv2
import random
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
from vm2m.network.module.aspp import ASPP
from vm2m.network.decoder import *
from vm2m.network.loss import LapLoss, loss_comp, loss_dtSSD, GradientLoss
from vm2m.network.backbone.resnet_enc import ResMaskEmbedShortCut_D
from vm2m.network.decoder.resnet_embed_atten_dec import ResShortCut_EmbedAtten_Dec
from vm2m.network.module.temporal_nn import TemporalNN
from vm2m.utils.utils import resizeAnyShape, compute_unknown

class MGM_Laplacian(nn.Module):
    def __init__(self, backbone, decoder, cfg):
        super(MGM_Laplacian, self).__init__()
        self.cfg = cfg

        self.encoder = backbone
        self.num_masks = cfg.backbone_args.num_mask

        self.aspp = ASPP(in_channel=512, out_channel=512)
        self.decoder = decoder
        if hasattr(self.encoder, 'mask_embed_layer'):
            if hasattr(self.decoder, 'temp_module_os16'):
                self.decoder.temp_module_os16.mask_embed_layer = self.encoder.mask_embed_layer

        # Some weights for loss
        self.loss_alpha_w = cfg.loss_alpha_w
        self.loss_alpha_grad_w = cfg.loss_alpha_grad_w
        self.loss_atten_w = cfg.loss_atten_w
        self.grad_loss = GradientLoss()

        self.train_temporal = False #cfg.decoder in ['res_shortcut_attention_spconv_temp_decoder_22']

        # For multi-inst loss
        self.loss_multi_inst_w = cfg.loss_multi_inst_w
        self.loss_multi_inst_warmup = cfg.loss_multi_inst_warmup
        if cfg.loss_multi_inst_type == 'l1':
            self.loss_multi_inst_func = F.l1_loss
        elif cfg.loss_multi_inst_type == 'l2':
            self.loss_multi_inst_func = F.mse_loss
        elif cfg.loss_multi_inst_type.startswith('smooth_l1'):
            beta = float(cfg.loss_multi_inst_type.split('_')[-1])
            self.loss_multi_inst_func = partial(F.smooth_l1_loss, beta=beta)


        need_init_weights = [self.aspp, self.decoder]

        # Init weights
        for module in need_init_weights:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        if self.train_temporal:
            self.freeze_to_train_temporal()
    
    def convert_syn_bn(self):
        self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.aspp = nn.SyncBatchNorm.convert_sync_batchnorm(self.aspp)
        self.decoder.convert_syn_bn()

    def freeze_to_train_temporal(self):
        # Freeze the encoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Freeze the ASPP
        self.aspp.eval()
        for param in self.aspp.parameters():
            param.requires_grad = False
        
        # Unfreeze the decoder
        self.decoder.train()
        for param in self.decoder.parameters():
            param.requires_grad = True

    
    def train(self, mode: bool = True):
        super().train(mode=mode)
        if mode and self.train_temporal:
            self.freeze_to_train_temporal()

    def forward(self, batch, return_ctx=False, mem_feat=[], mem_query=None, mem_details=None, **kwargs):
        '''
        x: b, n_f, 3, h, w, image tensors
        masks: b, n_frames, n_instances, h//8, w//8, coarse masks
        alphas: b, n_frames, n_instances, h, w, alpha matte
        trans_gt: b, n_frames, n_instances, h, w, incoherence mask ground truth
        '''
        x = batch['image']
        x_lap = batch['image_lap']
        masks = batch['mask']
        alphas = batch.get('alpha', None)

        # Combine input image and masks
        b, n_f, _, h, w = x.shape
        n_i = masks.shape[2]

        x = x.view(-1, 3, h, w)
        x_lap = x.view(-1, 3, h, w)
        if masks.shape[-1] != w:
            masks = masks.flatten(0,1)
            masks = F.interpolate(masks, size=(h, w), mode="nearest")
        else:
            masks = masks.view(-1, n_i, h, w)
        
        chosen_ids = None
        if self.num_masks > 0:
            inp_masks = masks
            if self.num_masks - n_i > 0:
                if not self.training:
                    padding = torch.zeros((b*n_f, self.num_masks - n_i, h, w), device=x.device)
                    inp_masks = torch.cat([masks, padding], dim=1)
                else:
                    # Pad randomly: input masks, trans_gt, alphas
                    chosen_ids = np.random.choice(self.num_masks, n_i, replace=False)
                    inp_masks = torch.zeros((b*n_f, self.num_masks, h, w), device=x.device)
                    inp_masks[:, chosen_ids, :, :] = masks
                    masks = inp_masks
                    if alphas is not None:
                        new_alphas = torch.zeros((b, n_f, self.num_masks, h, w), device=x.device)
                        new_alphas[:, :, chosen_ids, :, :] = alphas
                        alphas = new_alphas
                    if trans_gt is not None:
                        new_trans_gt = torch.zeros((b, n_f, self.num_masks, h, w), device=x.device)
                        new_trans_gt[:, :, chosen_ids, :, :] = trans_gt
                        trans_gt = new_trans_gt
                    n_i = self.num_masks

            inp = torch.cat([x, inp_masks], dim=1)
        else:
            inp = x

        if alphas is not None:
            alphas = alphas.view(-1, n_i, h, w)

        embedding, mid_fea = self.encoder(inp, masks=masks.reshape(b, n_f, n_i, h, w))
        embedding = self.aspp(embedding)
        
        pred = self.decoder(embedding, mid_fea, return_ctx=return_ctx, b=b, n_f=n_f, n_i=n_i, 
                            masks=masks, iter=batch.get('iter', 0), warmup_iter=self.cfg.mgm.warmup_iter, 
                            gt_alphas=alphas, mem_feat=mem_feat, mem_query=mem_query, mem_details=mem_details, lap_image=x_lap)
        
        output = {}
        if self.num_masks > 0 and self.training:
            output['alpha_os1'] = pred['alpha_os1'].view(b, n_f, self.num_masks, h, w)
            output['alpha_os4'] = pred['alpha_os4'].view(b, n_f, self.num_masks, h // 4, w // 4)
            output['alpha_os8'] = pred['alpha_os8'].view(b, n_f, self.num_masks, h // 8, w // 8)
        else:
            output['alpha_os1'] = pred['alpha_os1'][:, :n_i].view(b, n_f, n_i, h, w)
            output['alpha_os4'] = pred['alpha_os4'][:, :n_i].view(b, n_f, n_i, h // 4, w // 4)
            output['alpha_os8'] = pred['alpha_os8'][:, :n_i].view(b, n_f, n_i, h // 8, w // 8)

        if 'ctx' in pred:
            output['ctx'] = pred['ctx']
        
        # Fuse the output
        weight_os1, weight_os4, alpha_pred = pred.pop('weight_os1'), pred.pop('weight_os4'), pred.pop('alpha')
        
        # Reshape the output
        if self.num_masks > 0 and self.training:
            alpha_pred = alpha_pred.view(b, n_f, self.num_masks, h, w)
        else:
            alpha_pred = alpha_pred[:, :n_i].view(b, n_f, n_i, h, w)

        output['refined_masks'] = alpha_pred

        if self.training:
            alphas = alphas.view(-1, n_i, h, w)
            iter = batch['iter']
            
            # maskout padding masks
            valid_masks = alphas.sum((2, 3), keepdim=True) > 0
            valid_masks = valid_masks.float()
            for k, v in pred.items():
                if 'loss' in k or 'mem_' in k:
                    continue
                pred[k] = v * valid_masks

            loss_dict = self.compute_loss(pred, alpha_pred, alphas, weight_os1, weight_os4, iter, (b, n_f, self.num_masks, h, w))

            # Add loss max and min attention
            if 'loss_max_atten' in pred and self.loss_atten_w > 0:
                loss_dict['loss_max_atten'] = pred['loss_max_atten']
                loss_dict['total'] += loss_dict['loss_max_atten'] * self.loss_atten_w # + loss_dict['loss_min_atten']) * 0.1

            if not chosen_ids is None:
                for k, v in output.items():
                    output[k] = v[:, :, chosen_ids, :, :]
            return output, loss_dict

        # import pdb; pdb.set_trace()
        for k, v in output.items():
            output[k] = v[:, :, :n_i]
        for k in pred:
            if k.startswith("mem_"):
                output[k] = pred[k]
        return output
    
    def compute_weights(self, masks, k_size=30):
        h, w = masks.shape[-2:]
        ori_shape = masks.shape
        masks = masks.view(-1, 1, h, w)
        uncertain = (masks > 1.0/255.0) & (masks < 254.0/255.0)
        dilated_m = F.max_pool2d(uncertain.float(), kernel_size=k_size, stride=1, padding=k_size // 2)
        dilated_m = dilated_m[:,:, :h, :w]
        dilated_m = dilated_m.view(ori_shape)
        return dilated_m
    
    @staticmethod
    def regression_loss(logit, target, loss_type='l1', weight=None, topk=-1):
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
                loss = F.l1_loss(logit * weight, target * weight, reduction='none') 
                if topk > 0:
                    topk = int(weight.sum() * 0.5)
                    loss, _ = torch.topk(loss.view(-1), topk)
                    return loss.sum() / (topk + 1e-8)
                else:
                    return loss.sum() / (torch.sum(weight) + 1e-8)
            elif loss_type == 'l2':
                loss = F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
            else:
                raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
       
    def compute_loss(self, pred, fused_alpha_pred, alphas_gt, weight_os1, weight_os4, iter, alpha_shape):
        '''
        pred: dict of output from forward
        batch: dict of input batch
        '''
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        fused_alpha_pred = fused_alpha_pred.flatten(0, 1)

        # Compute alpha for each levels
        alpha_gt_os1 = alphas_gt
        alpha_gt_os4 = F.avg_pool2d(alphas_gt, kernel_size=4, stride=4)
        alpha_gt_os8 = F.avg_pool2d(alpha_gt_os4, kernel_size=2, stride=2)

        # Compute weights
        weight_os8 = torch.ones_like(alpha_pred_os8)
        valid_mask = alphas_gt.sum((2, 3), keepdim=True) > 0
        weight_os8 = weight_os8 * valid_mask

        weight_fuse = torch.ones_like(weight_os1)
        weight_fuse = weight_fuse * valid_mask

        if iter < self.cfg.mgm.warmup_iter or (iter < self.cfg.mgm.warmup_iter * 3 and random.randint(0,1) == 0):
            # Compute weight os4 and os1 from alphas
            temp_gt = F.interpolate(alpha_gt_os8, size=alpha_gt_os4.shape[-2:], mode='nearest')
            weight_os4 = compute_unknown(temp_gt, k_size=8)
            temp_gt = alpha_gt_os4 * weight_os4 + temp_gt * (1 - weight_os4)

            # Upsample and replace by OS1
            temp_gt = F.interpolate(temp_gt, size=alpha_gt_os1.shape[-2:], mode='nearest')
            weight_os1 = compute_unknown(temp_gt, k_size=15)
        
        loss_dict = {}
        

        # Add padding to alphas and trans_gt
        n_i = alphas_gt.shape[1]
        if self.num_masks - n_i > 0:
            padding = torch.zeros((alphas_gt.shape[0], self.num_masks - n_i, *alphas.shape[-2:]), device=alphas.device)
            alphas = torch.cat([alphas, padding], dim=1)

        # Reg loss
        total_loss = 0
        if self.loss_alpha_w > 0:
            logging.debug("Computing alpha loss")
            ref_alpha_os1 = self.regression_loss(alpha_pred_os1, alpha_gt_os1, loss_type=self.cfg.loss_alpha_type, weight=weight_os1)
            ref_alpha_os4 = self.regression_loss(alpha_pred_os4, alpha_gt_os4, loss_type=self.cfg.loss_alpha_type, weight=weight_os4)
            ref_alpha_os8 = self.regression_loss(alpha_pred_os8, alpha_gt_os8, loss_type=self.cfg.loss_alpha_type, weight=weight_os8)
            ref_alpha_fuse = self.regression_loss(fused_alpha_pred, alphas_gt, loss_type=self.cfg.loss_alpha_type, weight=weight_fuse)
            ref_alpha_loss = (ref_alpha_fuse + ref_alpha_os1 * 2 + ref_alpha_os4 * 2 + ref_alpha_os8) / 6.0
            loss_dict['loss_rec_os1'] = ref_alpha_os1
            loss_dict['loss_rec_os4'] = ref_alpha_os4
            loss_dict['loss_rec_os8'] = ref_alpha_os8
            loss_dict['loss_rec_fuse'] = ref_alpha_fuse
            loss_dict['loss_rec'] = ref_alpha_loss
            total_loss += ref_alpha_loss * self.loss_alpha_w
        
        if self.loss_alpha_grad_w > 0:
            grad_loss_os1 = self.grad_loss(alpha_pred_os1, alpha_gt_os1, weight_os1)
            grad_loss_os4 = self.grad_loss(alpha_pred_os4, alpha_gt_os4, weight_os4)
            grad_loss_os8 = self.grad_loss(alpha_pred_os8, alpha_gt_os8, weight_os8)
            grad_loss_fuse = self.grad_loss(fused_alpha_pred, alphas_gt, weight_fuse)
            grad_loss_all = (grad_loss_fuse + grad_loss_os1 * 2 + grad_loss_os4 * 2 + grad_loss_os8) / 6.0
            loss_dict['loss_grad_os1'] = grad_loss_os1
            loss_dict['loss_grad_os4'] = grad_loss_os4
            loss_dict['loss_grad_os8'] = grad_loss_os8
            loss_dict['loss_grad_fuse'] = grad_loss_fuse
            loss_dict['loss_grad'] = grad_loss_all
            total_loss += grad_loss_all * self.loss_alpha_grad_w

        if self.loss_multi_inst_w > 0 and iter >= self.loss_multi_inst_warmup:
            logging.debug("Computing multi inst loss")

            alpha_multi = fused_alpha_pred * valid_mask
            pred = alpha_multi.sum(1)
            mask = (pred > 1.0).float()
            multi_inst_loss = self.loss_multi_inst_func((pred * mask), mask, reduction='none')
            multi_inst_loss = multi_inst_loss.sum() / (mask.sum() + 1e-6)
            loss_dict['loss_multi_inst'] = multi_inst_loss
            total_loss += multi_inst_loss * self.loss_multi_inst_w


        loss_dict['total'] = total_loss
        return loss_dict