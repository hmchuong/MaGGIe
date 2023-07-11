import logging
import numpy as np
import cv2
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from .module.aspp import ASPP
from .backbone import *
from .loss import LapLoss, loss_comp

Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]
def get_unknown_tensor_from_pred(pred, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W 
    N, C, H, W = pred.shape

    device = pred.device
    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred<1.0/255.0] = 0
    uncertain_area[pred>1-1.0/255.0] = 0

    for n in range(N):
        uncertain_area_ = uncertain_area[n,0,:,:] # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
        uncertain_area[n,0,:,:] = uncertain_area_

    weight = np.zeros_like(uncertain_area)
    weight[uncertain_area == 1] = 1
    weight = torch.from_numpy(weight).to(device)

    return weight

class MGM(nn.Module):
    def __init__(self, backbone, cfg):
        super(MGM, self).__init__()
        self.backbone = backbone
        self.cfg = cfg

        self.encoder = backbone

        self.aspp = ASPP(in_channel=512, out_channel=512)
        self.decoder = eval(cfg.mgm.decoder)()

        # Some weights for loss
        self.loss_alpha_w = cfg.loss_alpha_w
        self.loss_comp_w = cfg.loss_comp_w
        self.loss_alpha_lap_w = cfg.loss_alpha_lap_w
        self.lap_loss = LapLoss()
    
    def fushion(self, pred):
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        ### Progressive Refinement Module in MGMatting Paper
        alpha_pred = alpha_pred_os8.clone().detach()
        weight_os4 = get_unknown_tensor_from_pred(alpha_pred, rand_width=30, train_mode=self.training)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4> 0]
        weight_os1 = get_unknown_tensor_from_pred(alpha_pred, rand_width=15, train_mode=self.training)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1 > 0]

        return alpha_pred, weight_os4, weight_os1

    def forward(self, batch, return_ctx=False):
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

        # Combine input image and masks
        b, n_f, _, h, w = x.shape
        n_i = masks.shape[2]

        x = x.view(-1, 3, h, w)
        if masks.shape[-1] != w:
            masks = masks.view(-1, 1, h//8, w//8)
            masks = F.interpolate(masks, size=(h, w), mode="nearest")
        else:
            masks = masks.view(-1, 1, h, w)
            
        inp = torch.cat([x, masks], dim=1)
        if alphas is not None:
            alphas = alphas.view(-1, 1, h, w)
        if trans_gt is not None:
            trans_gt = trans_gt.view(-1, 1, h, w)
        if fg is not None:
            fg = fg.view(-1, 3, h, w)
        if bg is not None:
            bg = bg.view(-1, 3, h, w)
       
        embedding, mid_fea = self.encoder(inp)
        embedding = self.aspp(embedding)
        pred = self.decoder(embedding, mid_fea, return_ctx=return_ctx)
        
        # Fushion
        logging.debug("Doing fusion")
        alpha_pred, weight_os4, weight_os1 = self.fushion(pred)
        logging.debug("Fusion done")

        
        output = {}
        
        output['alpha_os1'] = pred['alpha_os1'].view(b, n_f, n_i, h, w)
        output['alpha_os4'] = pred['alpha_os4'].view(b, n_f, n_i, h, w)
        output['alpha_os8'] = pred['alpha_os8'].view(b, n_f, n_i, h, w)
        if 'ctx' in pred:
            output['ctx'] = pred['ctx']

        # Reshape the output
        alpha_pred = alpha_pred.view(b, n_f, n_i, h, w)
        output['refined_masks'] = alpha_pred

        if self.training:
            alphas = alphas.view(-1, 1, h, w)
            trans_gt = trans_gt.view(-1, 1, h, w)
            iter = batch['iter']
            logging.debug("Computing loss")
            loss_dict = self.compute_loss(pred, weight_os4, weight_os1, alphas, trans_gt, fg, bg, iter)
            logging.debug("Loss computed")
            return output, loss_dict

        return output

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
       
    def compute_loss(self, pred, weight_os4, weight_os1, alphas, trans_gt, fg, bg, iter):
        '''
        pred: dict of output from forward
        batch: dict of input batch
        '''
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

        # Reg loss
        total_loss = 0
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
        
        loss_dict['total'] = total_loss
        return loss_dict


