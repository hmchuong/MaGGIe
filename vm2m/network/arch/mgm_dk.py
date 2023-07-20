import logging
import numpy as np
import cv2
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from vm2m.network.module.dcn_idk import DCInstDynKernelGenerator
from .mgm import MGM

class MGM_DK(MGM):
    def __init__(self, backbone, decoder, cfg):
        super(MGM_DK, self).__init__(backbone, decoder, cfg)
        self.ik_generator = DCInstDynKernelGenerator(cfg.aspp.out_channels, 
                                                     cfg.dynamic_kernel.hidden_dim, 
                                                     0,
                                                     cfg.dynamic_kernel.out_pixeldecoder)

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

        # Combine input image and masks
        b, n_f, _, h, w = x.shape
        n_i = masks.shape[2]

        x = x.view(-1, 3, h, w)
        
        if alphas is not None:
            alphas = alphas.view(-1, 1, h, w)
        if trans_gt is not None:
            trans_gt = trans_gt.view(-1, 1, h, w)
        if fg is not None:
            fg = fg.view(-1, 3, h, w)
        if bg is not None:
            bg = bg.view(-1, 3, h, w)
       
        embedding, mid_fea = self.encoder(x)
        embedding = self.aspp(embedding)

        # Generate dynamic kernels
        dec_kernels = self.ik_generator(embedding, masks)
        dec_kernels = dec_kernels.reshape(b * n_f, n_i, -1)
        pred = self.decoder(embedding, mid_fea, dec_kernels)
        
        # Fushion
        alpha_pred, weight_os4, weight_os1 = self.fushion(pred)

        
        output = {}
        
        output['alpha_os1'] = pred['alpha_os1'].view(b, n_f, n_i, h, w)
        output['alpha_os4'] = pred['alpha_os4'].view(b, n_f, n_i, h, w)
        output['alpha_os8'] = pred['alpha_os8'].view(b, n_f, n_i, h, w)

        # Reshape the output
        alpha_pred = alpha_pred.view(b, n_f, n_i, h, w)
        output['refined_masks'] = alpha_pred

        if self.training:
            alphas = alphas.view(-1, 1, h, w)
            trans_gt = trans_gt.view(-1, 1, h, w)
            iter = batch['iter']
            loss_dict = self.compute_loss(pred, weight_os4, weight_os1, alphas, trans_gt, fg, bg, iter)
            return output, loss_dict

        return output