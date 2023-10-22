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

from .mgm import MGM


class MGM_TempSpar(MGM):

    def forward(self, batch, return_ctx=False, mem_feat=None, mem_query=None, mem_details=None, **kwargs):
        '''
        x: b, n_f, 3, h, w, image tensors
        masks: b, n_frames, n_instances, h//8, w//8, coarse masks
        alphas: b, n_frames, n_instances, h, w, alpha matte
        trans_gt: b, n_frames, n_instances, h, w, incoherence mask ground truth
        '''
        x = batch['image']
        masks = batch['mask']
        alphas = batch.get('alpha', None)
        weights = batch.get('weight', None)
        trans_gt = batch.get('transition', None)
        fg = batch.get('fg', None)
        bg = batch.get('bg', None)

        # Combine input image and masks
        b, n_f, _, h, w = x.shape
        n_i = masks.shape[2]

        x = x.view(-1, 3, h, w)
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
        if trans_gt is not None:
            trans_gt = trans_gt.view(-1, n_i, h, w)
        if fg is not None:
            fg = fg.view(-1, 3, h, w)
        if bg is not None:
            bg = bg.view(-1, 3, h, w)

        embedding, mid_fea = self.encoder(inp, masks=masks.reshape(b, n_f, n_i, h, w))
        embedding = self.aspp(embedding)
        
        pred = self.decoder(embedding, mid_fea, return_ctx=return_ctx, b=b, n_f=n_f, n_i=n_i, 
                            masks=masks, iter=batch.get('iter', 0), warmup_iter=self.cfg.mgm.warmup_iter, 
                            gt_alphas=alphas, mem_feat=mem_feat, mem_query=mem_query, mem_details=mem_details, spar_gt=trans_gt)
        pred_notemp = None
        if isinstance(pred, tuple):
            pred, pred_notemp = pred

        # Fushion
        if 'refined_masks' in pred:
            alpha_pred = pred.pop("refined_masks")
            weight_os4 = pred.pop("detail_mask").type(alpha_pred.dtype)
            weight_os1 = weight_os4
        else:
            alpha_pred, weight_os4, weight_os1 = self.fushion(pred)
        
        
        output = {}
        if self.num_masks > 0 and self.training:
            output['alpha_os1'] = pred['alpha_os1'].view(b, n_f, self.num_masks, h, w)
            output['alpha_os4'] = pred['alpha_os4'].view(b, n_f, self.num_masks, h, w)
            output['alpha_os8'] = pred['alpha_os8'].view(b, n_f, self.num_masks, h, w)
        else:
            output['alpha_os1'] = pred['alpha_os1'][:, :n_i].view(b, n_f, n_i, h, w)
            output['alpha_os4'] = pred['alpha_os4'][:, :n_i].view(b, n_f, n_i, h, w)
            output['alpha_os8'] = pred['alpha_os8'][:, :n_i].view(b, n_f, n_i, h, w)
        if 'ctx' in pred:
            output['ctx'] = pred['ctx']

        # Reshape the output
        if self.num_masks > 0 and self.training:
            alpha_pred = alpha_pred.view(b, n_f, self.num_masks, h, w)
        else:
            alpha_pred = alpha_pred[:, :n_i].view(b, n_f, n_i, h, w)
        
        output['refined_masks'] = alpha_pred
        # cv2.imwrite("test_a.png", (alpha_pred[0,0,7].detach().cpu().numpy() * 255).astype('uint8'))
        # import pdb; pdb.set_trace()
        diff_pred_forward = pred.pop('diff_forward', None)
        diff_pred_backward = pred.pop('diff_backward', None)
        temp_alpha = pred.pop('temp_alpha', None)

        if diff_pred_backward is not None:
            # Adding diff_pred and temp_alpha for visualization
            diff_pred_backward = diff_pred_backward.repeat(1, 1, n_i, 1, 1)
            diff_pred_forward = diff_pred_forward.repeat(1, 1, n_i, 1, 1)
            # temp_alpha = temp_alpha[:, None].repeat(1, n_f, 1, 1, 1)
            output['diff_pred_backward'] = diff_pred_backward
            output['diff_pred_forward'] = diff_pred_forward
            output['temp_alpha'] = temp_alpha

        if self.training:
            alphas = alphas.view(-1, n_i, h, w)
            trans_gt = trans_gt.view(-1, n_i, h, w)
            if weights is not None:
                weights = weights.view(-1, n_i, h, w)
            iter = batch['iter']
            
            # maskout padding masks
            valid_masks = alphas.sum((2, 3), keepdim=True) > 0
            valid_masks = valid_masks.float()
            for k, v in pred.items():
                if 'loss' in k or 'mem_' in k:
                    continue
                # import pdb; pdb.set_trace()
                pred[k] = v * valid_masks

            loss_dict = self.compute_loss(pred, weight_os4, weight_os1, weights, alphas, trans_gt, fg, bg, iter, (b, n_f, self.num_masks, h, w))
    
            if 'loss_temp' in pred:
                loss_dict['loss_temp_bce'] = pred['loss_temp_bce']
                loss_dict['loss_temp'] = pred['loss_temp']
                loss_dict['total'] += pred['loss_temp']
            if 'loss_temp_fusion' in pred:
                loss_dict['loss_temp_fusion'] = pred['loss_temp_fusion']
            if 'loss_temp_dtssd' in pred:
                loss_dict['loss_temp_dtssd'] = pred['loss_temp_dtssd']

            # Add loss max and min attention
            if 'loss_max_atten' in pred and self.loss_atten_w > 0:
                loss_dict['loss_max_atten'] = pred['loss_max_atten']
                loss_dict['total'] += loss_dict['loss_max_atten'] * self.loss_atten_w # + loss_dict['loss_min_atten']) * 0.1

            if not chosen_ids is None:
                for k, v in output.items():
                    output[k] = v[:, :, chosen_ids, :, :]
            
            return output, loss_dict

        for k, v in output.items():
            output[k] = v[:, :, :n_i]
        
        for k in pred:
            if k.startswith("mem_"):
                output[k] = pred[k]
        
        return output