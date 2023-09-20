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
    def __init__(self, backbone, decoder, cfg):
        super(MGM, self).__init__()
        self.cfg = cfg

        self.encoder = backbone
        self.num_masks = cfg.backbone_args.num_mask

        self.aspp = ASPP(in_channel=512, out_channel=512)
        self.decoder = decoder
        if hasattr(self.encoder, 'mask_embed_layer'):
            if hasattr(self.decoder, 'temp_module_os16'):
                self.decoder.temp_module_os16.mask_embed_layer = self.encoder.mask_embed_layer
        # if isinstance(self.encoder, ResMaskEmbedShortCut_D) and isinstance(self.decoder, ResShortCut_EmbedAtten_Dec):
        #     self.decoder.refine_OS1.id_embedding = self.encoder.mask_embed
        #     self.decoder.refine_OS4.id_embedding = self.encoder.mask_embed
        #     self.decoder.refine_OS8.id_embedding = self.encoder.mask_embed

        # Some weights for loss
        self.loss_alpha_w = cfg.loss_alpha_w
        self.loss_comp_w = cfg.loss_comp_w
        self.loss_alpha_lap_w = cfg.loss_alpha_lap_w
        self.loss_dtSSD_w = cfg.loss_dtSSD_w
        self.loss_alpha_grad_w = cfg.loss_alpha_grad_w
        self.loss_atten_w = cfg.loss_atten_w
        self.lap_loss = LapLoss()
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
        
        # Train the temporal module
        # self.decoder.temp_module.train()
        # for param in self.decoder.temp_module.parameters():
        #     param.requires_grad = True

    
    def train(self, mode: bool = True):
        super().train(mode=mode)
        if mode and self.train_temporal:
            self.freeze_to_train_temporal()


    def fushion(self, pred):
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        ### Progressive Refinement Module in MGMatting Paper
        alpha_pred = alpha_pred_os8.clone().detach()
        weight_os4 = get_unknown_tensor_from_pred(alpha_pred, rand_width=30, train_mode=self.training)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4> 0]
        weight_os1 = get_unknown_tensor_from_pred(alpha_pred, rand_width=15, train_mode=self.training)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1 > 0]

        return alpha_pred, weight_os4, weight_os1

    def forward(self, batch, return_ctx=False, mem_feat=[], mem_query=None, mem_details=None, **kwargs):
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

        if self.train_temporal:
            with torch.no_grad():
                embedding, mid_fea = self.encoder(inp, masks=masks.reshape(b, n_f, n_i, h, w))
                embedding = self.aspp(embedding)
        else:
            # import pdb; pdb.set_trace()
            embedding, mid_fea = self.encoder(inp, masks=masks.reshape(b, n_f, n_i, h, w))
            embedding = self.aspp(embedding)
        
        pred = self.decoder(embedding, mid_fea, return_ctx=return_ctx, b=b, n_f=n_f, n_i=n_i, 
                            masks=masks, iter=batch.get('iter', 0), warmup_iter=self.cfg.mgm.warmup_iter, 
                            gt_alphas=alphas, mem_feat=mem_feat, mem_query=mem_query, mem_details=mem_details)
        
        # Fushion
        if 'refined_masks' in pred:
            alpha_pred = pred.pop("refined_masks")
            weight_os4 = pred.pop("weight_os4")
            weight_os1 = pred.pop("weight_os1")
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

        if self.training:
            alphas = alphas.view(-1, n_i, h, w)
            trans_gt = trans_gt.view(-1, n_i, h, w)
            if weights is not None:
                weights = weights.view(-1, n_i, h, w)
            iter = batch['iter']
            
            # maskout padding masks
            valid_masks = trans_gt.sum((2, 3), keepdim=True) > 0
            valid_masks = valid_masks.float()
            for k, v in pred.items():
                if 'loss' in k or 'mem_' in k:
                    continue
                pred[k] = v * valid_masks

            loss_dict = self.compute_loss(pred, weight_os4, weight_os1, weights, alphas, trans_gt, fg, bg, iter, (b, n_f, self.num_masks, h, w))

            # Add loss max and min attention
            if 'loss_max_atten' in pred and self.loss_atten_w > 0:
                loss_dict['loss_max_atten'] = pred['loss_max_atten']
                # loss_dict['loss_min_atten'] = pred['loss_min_atten']
                loss_dict['total'] += loss_dict['loss_max_atten'] * self.loss_atten_w # + loss_dict['loss_min_atten']) * 0.1

            if not chosen_ids is None:
                for k, v in output.items():
                    output[k] = v[:, :, chosen_ids, :, :]
            return output, loss_dict

        # import pdb; pdb.set_trace()
        for k, v in output.items():
            output[k] = v[:, :, :n_i]
        # if 'embedding' in pred:
            # output['embedding'] = pred['embedding']
        # if 'embedding_os8' in pred:
            # output['embedding'] = {'os32': pred['embedding_os32'], 'os8': pred['embedding_os8']}
            # output['embedding'] = {'os8': pred['embedding_os8']}
        # output['prev_mask'] = alpha_pred
        for k in pred:
            if k.startswith("mem_"):
                output[k] = pred[k]
        return output

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
       
    def compute_loss(self, pred, weight_os4, weight_os1, correct_weights, alphas, trans_gt, fg, bg, iter, alpha_shape):
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
        valid_mask = alphas.sum((2, 3), keepdim=True) > 0
        weight_os8 = weight_os8 * valid_mask

        if correct_weights is not None:
            weight_os1 = weight_os1 * correct_weights
            weight_os4 = weight_os4 * correct_weights
            weight_os8 = weight_os8 * correct_weights

        # Add padding to alphas and trans_gt
        n_i = alphas.shape[1]
        if self.num_masks - n_i > 0:
            padding = torch.zeros((alphas.shape[0], self.num_masks - n_i, *alphas.shape[-2:]), device=alphas.device)
            alphas = torch.cat([alphas, padding], dim=1)
            trans_gt = torch.cat([trans_gt, padding], dim=1)

        # Reg loss
        total_loss = 0
        if self.loss_alpha_w > 0:
            logging.debug("Computing alpha loss")
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
            logging.debug("Computing lap loss")
            h, w = alpha_pred_os1.shape[-2:]
            lap_loss_os1 = self.lap_loss(alpha_pred_os1.view(-1, 1, h, w), alphas.view(-1, 1, h, w), weight_os1.view(-1, 1, h, w))
            lap_loss_os4 = self.lap_loss(alpha_pred_os4.view(-1, 1, h, w), alphas.view(-1, 1, h, w), weight_os4.view(-1, 1, h, w))
            lap_loss_os8 = self.lap_loss(alpha_pred_os8.view(-1, 1, h, w), alphas.view(-1, 1, h, w), weight_os8.view(-1, 1, h, w))
            lap_loss = (lap_loss_os1 * 2 + lap_loss_os4 * 1 + lap_loss_os8 * 1) / 5.0
            loss_dict['loss_lap_os1'] = lap_loss_os1
            loss_dict['loss_lap_os4'] = lap_loss_os4
            loss_dict['loss_lap_os8'] = lap_loss_os8
            loss_dict['loss_lap'] = lap_loss
            total_loss += lap_loss * self.loss_alpha_lap_w
        
        if self.loss_alpha_grad_w > 0:
            grad_loss_os1 = self.grad_loss(alpha_pred_os1, alphas, weight_os1)
            grad_loss_os4 = self.grad_loss(alpha_pred_os4, alphas, weight_os4)
            grad_loss_os8 = self.grad_loss(alpha_pred_os8, alphas, weight_os8)
            grad_loss = (grad_loss_os1 * 2 + grad_loss_os4 * 1 + grad_loss_os8 * 1) / 5.0
            loss_dict['loss_grad_os1'] = grad_loss_os1
            loss_dict['loss_grad_os4'] = grad_loss_os4
            loss_dict['loss_grad_os8'] = grad_loss_os8
            loss_dict['loss_grad'] = grad_loss
            total_loss += grad_loss * self.loss_alpha_grad_w

        

        if self.loss_multi_inst_w > 0 and iter >= self.loss_multi_inst_warmup:
            logging.debug("Computing multi inst loss")
            # multi_inst_os1 = self.regression_loss(alpha_pred_os1.sum(1), alphas.sum(1), loss_type=self.cfg.loss_alpha_type)
            # multi_inst_os4 = self.regression_loss(alpha_pred_os4.sum(1), alphas.sum(1), loss_type=self.cfg.loss_alpha_type)
            alpha_multi_os8 = alpha_pred_os8 * valid_mask
            pred = alpha_multi_os8.sum(1)
            mask = (pred > 1.0).float()
            # multi_inst_os8 = F.l1_loss((pred * mask), mask, reduction='sum')
            # multi_inst_os8 = F.smooth_l1_loss((pred * mask), mask, reduction='none', beta=0.5)
            multi_inst_os8 = self.loss_multi_inst_func((pred * mask), mask, reduction='none')
            multi_inst_os8 = multi_inst_os8.sum() / (mask.sum() + 1e-6)
            # print(multi_inst_os8)
            # multi_inst_loss = (multi_inst_os1 * 2 + multi_inst_os4 * 1 + multi_inst_os8 * 1) / 5.0
            # loss_dict['loss_multi_inst_os1'] = multi_inst_os1
            # loss_dict['loss_multi_inst_os4'] = multi_inst_os4
            # loss_dict['loss_multi_inst_os8'] = multi_inst_os8
            loss_dict['loss_multi_inst'] = multi_inst_os8
            total_loss += multi_inst_os8 * self.loss_multi_inst_w

        if self.loss_dtSSD_w > 0:
            alpha_pred_os8 = alpha_pred_os8.reshape(*alpha_shape)
            alpha_pred_os4 = alpha_pred_os4.reshape(*alpha_shape)
            alpha_pred_os1 = alpha_pred_os1.reshape(*alpha_shape)
            alphas = alphas.reshape(*alpha_shape)
            trans_gt = trans_gt.reshape(*alpha_shape)
            dtSSD_loss_os1 = loss_dtSSD(alpha_pred_os1, alphas, trans_gt)
            dtSSD_loss_os4 = loss_dtSSD(alpha_pred_os4, alphas, trans_gt)
            dtSSD_loss_os8 = loss_dtSSD(alpha_pred_os8, alphas, trans_gt)
            dtSSD_loss = (dtSSD_loss_os1 * 2 + dtSSD_loss_os4 * 1 + dtSSD_loss_os8 * 1) / 5.0
            loss_dict['loss_dtSSD_os1'] = dtSSD_loss_os1
            loss_dict['loss_dtSSD_os4'] = dtSSD_loss_os4
            loss_dict['loss_dtSSD_os8'] = dtSSD_loss_os8
            loss_dict['loss_dtSSD'] = dtSSD_loss
            total_loss += dtSSD_loss * self.loss_dtSSD_w

        loss_dict['total'] = total_loss
        return loss_dict


class MGM_SingInst(MGM):
    def forward(self, batch, **kwargs):
        if self.training:
            return super().forward(batch, **kwargs)
        masks = batch['mask']
        n_i = masks.shape[2]
        # if self.num_masks == 1:
        outputs = []
        # interate one mask at a time
        batch = copy.deepcopy(batch)
        for i in range(n_i):
            batch['mask'] = masks[:, :, i:i+1]
            outputs.append(super().forward(batch, **kwargs))
        for k in outputs[0].keys():
            outputs[0][k] = torch.cat([o[k] for o in outputs], 2)
        return outputs[0]
        # return super().forward(batch, return_ctx)