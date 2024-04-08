from functools import partial
import logging
import numpy as np
import cv2
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..module import ASPP
from ..loss import LapLoss, loss_dtSSD, GradientLoss
from ...utils.utils import compute_unknown

class MaGGIe(nn.Module):
    def __init__(self, encoder, decoder, cfg):
        super(MaGGIe, self).__init__()
        self.cfg = cfg
        self.num_masks = cfg.backbone_args.num_mask

        self.encoder = encoder
        
        self.aspp = ASPP(in_channel=cfg.aspp.in_channels, out_channel=cfg.aspp.out_channels)
        self.decoder = decoder

        # Some weights for loss
        self.loss_alpha_w = cfg.loss_alpha_w
        self.loss_alpha_lap_w = cfg.loss_alpha_lap_w
        self.loss_alpha_grad_w = cfg.loss_alpha_grad_w
        self.loss_atten_w = cfg.loss_atten_w
        self.reweight_os8 = cfg.loss_reweight_os8
        self.loss_dtSSD_w = cfg.loss_dtSSD_w
        
        self.lap_loss = LapLoss()
        self.grad_loss = GradientLoss()

        need_init_weights = [self.aspp, self.decoder]

        # Init weights
        for module in need_init_weights:
            for name, p in module.named_parameters():
                if "context_token" in name:
                    continue
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def fuse(self, pred):
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        ### Progressive Refinement Module in MGMatting Paper
        alpha_pred = alpha_pred_os8.clone()
        weight_os4 = compute_unknown(alpha_pred, k_size=30, train_mode=self.training)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4> 0]
        weight_os1 = compute_unknown(alpha_pred, k_size=15, train_mode=self.training)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1 > 0]

        return alpha_pred, weight_os4, weight_os1

    def forward(self, batch, **kwargs):
        '''
        batch:
            image: b, n_f, 3, h, w 
                image tensors
            mask: b, n_f, n_i, h//8, w//8
                coarse masks
            alpha: b, n_f, n_i, h, w
                GT alpha matte
            transition: b, n_f, n_i, h, w, 
                GT unknown mask
            fg: b, n_f, 3, h, w
                foreground image
            bg: b, n_f, 3, h, w
                background image
        '''

        # Forward encoder
        masks, alphas, trans_gt, b, n_f, h, w, n_i, chosen_ids, embedding, mid_fea = self.forward_encoder(batch)
        
        # Forward through decoder
        pred = self.decoder(embedding, mid_fea, b=b, n_f=n_f, n_i=n_i, masks=masks, iter=batch.get('iter', 0), gt_alphas=alphas, spar_gt=trans_gt, **kwargs)
        
        if isinstance(pred, tuple):
            pred = pred[0]

        # Fusion
        weight_os1, weight_os4 = None, None
        if 'refined_masks' in pred:
            alpha_pred = pred.pop("refined_masks")
            if 'detail_mask' in pred:
                weight_os4 = pred["detail_mask"].type(alpha_pred.dtype)
                weight_os1 = weight_os4
        else:
            alpha_pred, weight_os4, weight_os1 = self.fuse(pred)
        
        # During training: 75% use the weight os4 and os1 masks, 25% use the detail mask
        if self.training and 'weight_os4' in pred and np.random.rand() < 0.75:
            weight_os4 = pred.pop("weight_os4")
            weight_os1 = pred.pop("weight_os1")
        
        # Reshape output back to 5D tensors
        output = self.transform_output(b, n_f, h, w, n_i, pred, alpha_pred)

        # Compute loss during training
        if self.training:
            alphas = alphas.view(-1, n_i, h, w)
            trans_gt = trans_gt.view(-1, n_i, h, w)
            
            # maskout padding masks
            valid_masks = trans_gt.sum((2, 3), keepdim=True) > 0
            valid_masks = valid_masks.float()
            for k, v in pred.items():
                if 'loss' in k or 'mem_' in k:
                    continue
                pred[k] = v * valid_masks

            loss_dict = self.compute_loss(pred, weight_os4, weight_os1, alphas, trans_gt, (b, n_f, self.num_masks, h, w), reweight_os8=self.reweight_os8)

            # Add attention loss from decoder
            self.update_additional_decoder_loss(pred, loss_dict)

            # Ignore random padding positions
            if not chosen_ids is None:
                for k, v in output.items():
                    output[k] = v[:, :, chosen_ids, :, :]
            return output, loss_dict

        # During inference, remove padding instances
        for k, v in output.items():
            output[k] = v[:, :, :n_i]
        
        # Update mem to output
        for k in pred:
            if k.startswith("mem_"):
                output[k] = pred[k]
                
        return output

    def update_additional_decoder_loss(self, pred, loss_dict):
        if 'loss_max_atten' in pred and self.loss_atten_w > 0:
            loss_dict['loss_max_atten'] = pred['loss_max_atten']
            loss_dict['total'] += loss_dict['loss_max_atten'] * self.loss_atten_w

    def transform_output(self, b, n_f, h, w, n_i, pred, alpha_pred):
        output = {}
        n_out_inst = self.num_masks if (self.training and self.num_masks > 0) else n_i
        if 'alpha_os1' in pred:
            output['alpha_os1'] = pred['alpha_os1'][:, :n_out_inst].view(b, n_f, n_out_inst, h, w)
            output['alpha_os4'] = pred['alpha_os4'][:, :n_out_inst].view(b, n_f, n_out_inst, h, w)
        
        output['alpha_os8'] = pred['alpha_os8'][:, :n_out_inst].view(b, n_f, n_out_inst, h, w)
        alpha_pred = alpha_pred[:, :n_out_inst].view(b, n_f, n_out_inst, h, w)
        output['refined_masks'] = alpha_pred

        if 'detail_mask' in pred:
            output['detail_mask'] = pred['detail_mask'][:, :n_out_inst].view(b, n_f, n_out_inst, h, w)
        return output

    def forward_encoder(self, batch):
        x = batch['image']
        masks = batch['mask']
        alphas = batch.get('alpha', None)
        trans_gt = batch.get('transition', None) # Unknown region ground truth
        fg = batch.get('fg', None)
        bg = batch.get('bg', None)

        # Get shape information
        b, n_f, _, h, w = x.shape
        n_i = masks.shape[2]

        x = x.view(-1, 3, h, w)
        
        # Resize masks
        if masks.shape[-1] != w:
            masks = masks.flatten(0,1)
            masks = F.interpolate(masks, size=(h, w), mode="nearest")
        else:
            masks = masks.view(-1, n_i, h, w)

        # Prepare input
        masks, alphas, trans_gt, n_i, chosen_ids, inp = self.prepare_input(x, masks, alphas, trans_gt, b, n_f, h, w, n_i)

        # Reshape the input
        if alphas is not None:
            alphas = alphas.view(-1, n_i, h, w)
        if trans_gt is not None:
            trans_gt = trans_gt.view(-1, n_i, h, w)
        if fg is not None:
            fg = fg.view(-1, 3, h, w)
        if bg is not None:
            bg = bg.view(-1, 3, h, w)

        # Forward through encoder and ASPP
        embedding, mid_fea = self.encoder(inp, masks=masks.reshape(b, n_f, n_i, h, w))
        embedding = self.aspp(embedding)
        return masks,alphas,trans_gt,b,n_f,h,w,n_i,chosen_ids,embedding,mid_fea

    def prepare_input(self, x, masks, alphas, trans_gt, b, n_f, h, w, n_i):
        chosen_ids = None
        if self.num_masks > 0:
            # If we have mask guidance
            inp_masks = masks

            if self.num_masks - n_i > 0:
                # If we need to pad empty masks (for multi-instance forward)
                if not self.training:
                    # During inference, pad with zeros at the end
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
            
            # Stack image and guidance masks
            inp = torch.cat([x, inp_masks], dim=1)
        else:
            inp = x
        return masks, alphas, trans_gt, n_i, chosen_ids, inp

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
    
    def compute_loss(self, pred, weight_os4, weight_os1, alphas, trans_gt, alpha_shape, reweight_os8=True):

        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred.get('alpha_os1', None), pred.get('alpha_os4', None), pred['alpha_os8']

        loss_dict = {}
        weight_os8 = torch.ones_like(alpha_pred_os8)
        valid_mask = alphas.sum((2, 3), keepdim=True) > 0
        weight_os8 = weight_os8 * valid_mask

        # Using reweighting
        if reweight_os8:
            unknown_gt = (alphas <= 254.0/255.0) & (alphas >= 1.0/255.0)
            unknown_pred_os8 = (alpha_pred_os8 <= 254.0/255.0) & (alpha_pred_os8 >= 1.0/255.0)
            weight_os8 = (unknown_gt | unknown_pred_os8).type(weight_os8.dtype) + weight_os8

        # Add padding to alphas and trans_gt
        n_i = alphas.shape[1]
        if self.num_masks - n_i > 0:
            padding = torch.zeros((alphas.shape[0], self.num_masks - n_i, *alphas.shape[-2:]), device=alphas.device)
            alphas = torch.cat([alphas, padding], dim=1)
            trans_gt = torch.cat([trans_gt, padding], dim=1)
       
        # Reconstruction loss
        total_loss = 0
        if self.loss_alpha_w > 0:
            ref_alpha_loss = 0
            
            if alpha_pred_os1 is not None:
                
                ref_alpha_os1 = self.regression_loss(alpha_pred_os1, alphas, loss_type=self.cfg.loss_alpha_type, weight=weight_os1)
                ref_alpha_os4 = self.regression_loss(alpha_pred_os4, alphas, loss_type=self.cfg.loss_alpha_type, weight=weight_os4)
                ref_alpha_os8 = self.regression_loss(alpha_pred_os8, alphas, loss_type=self.cfg.loss_alpha_type, weight=weight_os8)
                
                ref_alpha_loss += ref_alpha_os1 * 2 + ref_alpha_os4 + ref_alpha_os8
                
                loss_dict['loss_rec_os1'] = ref_alpha_os1
                loss_dict['loss_rec_os4'] = ref_alpha_os4
                loss_dict['loss_rec_os8'] = ref_alpha_os8

            loss_dict['loss_rec'] = ref_alpha_loss
            total_loss += ref_alpha_loss * self.loss_alpha_w

        # Lap loss
        if self.loss_alpha_lap_w > 0:
            logging.debug("Computing lap loss")
            h, w = alpha_pred_os8.shape[-2:]
            lap_loss = 0
            
            if alpha_pred_os1 is not None:
                lap_loss_os1 = self.lap_loss(alpha_pred_os1.view(-1, 1, h, w), alphas.view(-1, 1, h, w), weight_os1.view(-1, 1, h, w))
                lap_loss_os4 = self.lap_loss(alpha_pred_os4.view(-1, 1, h, w), alphas.view(-1, 1, h, w), weight_os4.view(-1, 1, h, w))
                lap_loss_os8 = self.lap_loss(alpha_pred_os8.view(-1, 1, h, w), alphas.view(-1, 1, h, w), weight_os8.view(-1, 1, h, w))
            
                loss_dict['loss_lap_os1'] = lap_loss_os1
                loss_dict['loss_lap_os4'] = lap_loss_os4
                loss_dict['loss_lap_os8'] = lap_loss_os8
            
                lap_loss += lap_loss_os1 * 2 + lap_loss_os4  + lap_loss_os8

            loss_dict['loss_lap'] = lap_loss
            total_loss += lap_loss * self.loss_alpha_lap_w
        
        # Gradient loss
        if self.loss_alpha_grad_w > 0:
            grad_loss = 0
            
            if alpha_pred_os1 is not None:
            
                grad_loss_os1 = self.grad_loss(alpha_pred_os1, alphas, weight_os1)
                grad_loss_os4 = self.grad_loss(alpha_pred_os4, alphas, weight_os4)
                grad_loss_os8 = self.grad_loss(alpha_pred_os8, alphas, weight_os8)
                grad_loss += grad_loss_os1 * 2 + grad_loss_os4 + grad_loss_os8
            
                loss_dict['loss_grad_os1'] = grad_loss_os1
                loss_dict['loss_grad_os4'] = grad_loss_os4
                loss_dict['loss_grad_os8'] = grad_loss_os8
            
            loss_dict['loss_grad'] = grad_loss
            total_loss += grad_loss * self.loss_alpha_grad_w

        # dtSSD loss, only for video
        if self.loss_dtSSD_w > 0:
            alpha_pred_os8 = alpha_pred_os8.reshape(*alpha_shape)
            alpha_pred_os4 = alpha_pred_os4.reshape(*alpha_shape)
            alpha_pred_os1 = alpha_pred_os1.reshape(*alpha_shape)
            alphas = alphas.reshape(*alpha_shape)

            dtSSD_loss_os1 = loss_dtSSD(alpha_pred_os1, alphas, weight_os1.reshape(*alpha_shape))
            dtSSD_loss_os4 = loss_dtSSD(alpha_pred_os4, alphas, weight_os4.reshape(*alpha_shape))
            dtSSD_loss_os8 = loss_dtSSD(alpha_pred_os8, alphas, weight_os8.reshape(*alpha_shape))
            dtSSD_loss = dtSSD_loss_os1 * 2 + dtSSD_loss_os4 * 1 + dtSSD_loss_os8 * 1
            
            loss_dict['loss_dtSSD_os1'] = dtSSD_loss_os1
            loss_dict['loss_dtSSD_os4'] = dtSSD_loss_os4
            loss_dict['loss_dtSSD_os8'] = dtSSD_loss_os8
            loss_dict['loss_dtSSD'] = dtSSD_loss
            
            total_loss += dtSSD_loss * self.loss_dtSSD_w

        loss_dict['total'] = total_loss
        return loss_dict

class MGM_SingInst(MaGGIe):
    def forward(self, batch, **kwargs):
        if self.training:
            return super().forward(batch, **kwargs)
        masks = batch['mask']
        n_i = masks.shape[2]
        outputs = []
        # interate one mask at a time
        batch = copy.deepcopy(batch)
        for i in range(n_i):
            batch['mask'] = masks[:, :, i:i+1]
            outputs.append(super().forward(batch, **kwargs))
        for k in outputs[0].keys():
            outputs[0][k] = torch.cat([o[k] for o in outputs], 2)
        return outputs[0]