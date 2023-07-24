import cv2
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from vm2m.network.module.shm import SHM
from vm2m.network.loss import LapLoss, loss_comp
from .mgm import MGM


def _upsample_like(src,tar,mode='bilinear'):
    src = F.interpolate(src,size=tar.shape[-2:],mode=mode,align_corners=False if mode=='bilinear' else None)
    return src
upas = _upsample_like

def reshape5D(x, scale_factor=0.5, multiply_by=64):
    shape = x.shape
    dtype = x.dtype
    x = x.view(-1, shape[-3], *shape[-2:]).float()
    x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    h_pad = (multiply_by - x.shape[-2] % multiply_by) % multiply_by
    w_pad = (multiply_by - x.shape[-1] % multiply_by) % multiply_by
    x = F.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
    x = x.view(*shape[:-2], *x.shape[-2:]).to(dtype)
    return x

def batch_slice(tensor, pos, size, mode='bilinear'):
    n, c, h, w = tensor.shape
    patchs = []
    for i in range(n):
        # x1, y1, x2, y2 = torch.clamp(pos[i], 0, 1)
        x1, y1, x2, y2 = pos[i]
        x1 = int(x1.item() * w)
        y1 = int(y1.item() * h)
        x2 = int(x2.item() * w)
        y2 = int(y2.item() * h)
        patch = tensor[i:i+1, :, y1:y2, x1:x2].contiguous()
        patch = F.interpolate(patch, (size[0], size[1]), align_corners=False if mode=='bilinear' else None, mode=mode)
        patchs.append(patch)
    return torch.cat(patchs, dim=0)

class SparseMat(nn.Module):
    def __init__(self, backbone, decoder, cfg):
        super(SparseMat, self).__init__()
        self.cfg = cfg
        self.mgm = MGM(backbone, decoder, cfg)
        self.shm = SHM(inc=4)
        self.lr_scale = cfg.shm.lr_scale
        self.stride = cfg.shm.dilation_kernel
        self.dilate_op = nn.MaxPool2d(self.stride, stride=1, padding=self.stride//2)
        self.max_n_pixel = cfg.shm.max_n_pixel

        self.loss_alpha_w = cfg.loss_alpha_w
        self.loss_comp_w = cfg.loss_comp_w
        self.loss_alpha_lap_w = cfg.loss_alpha_lap_w
        self.lap_loss = LapLoss()

        state_dict = torch.load(cfg.shm.mgm_weights, map_location='cpu')
        self.mgm.load_state_dict(state_dict, strict=True)

        self.mgm.eval()
        for p in self.mgm.parameters():
            p.requires_grad = False
    
    def train(self, mode=True):
        super(SparseMat, self).train(mode)
        self.mgm.eval()

    @torch.no_grad()
    def generate_sparse_inputs(self, img, lr_pred, mask):
        lr_pred = (lr_pred - 0.5) / 0.5
        x = torch.cat((img, lr_pred), dim=1)
        indices = torch.where(mask.squeeze(1)>0)

        if self.training and len(indices[0]) > 1600000:
            ids = torch.randperm(len(indices[0]))[:1600000]
            indices = [i[ids] for i in indices]
        
        x = x.permute(0,2,3,1)
        x = x[indices]
        indices = torch.stack(indices, dim=1)
        return x, indices

    def dilate(self, alpha, stride=15):
        mask = torch.logical_and(alpha>0.01, alpha<0.99).float()
        mask = self.dilate_op(mask)
        return mask

    def gen_lr_batch(self, batch, scale=0.5):
        lr_batch = {}
        lr_batch['image'] = reshape5D(batch['image'], scale_factor=scale, multiply_by=64)
        lr_batch['mask'] = reshape5D(batch['mask'], scale_factor=4.0, multiply_by=64)
        return lr_batch

    def forward_inference(self, lr_pred, x_hr, ctx, bs, n_f):
        
        mask, mask_s, mask_t, shared = self.generate_sparsity_map(lr_pred[1:], x_hr[1:], x_hr[:-1])
        n_pixel = mask.sum().item()
        pre_mask = self.dilate(lr_pred[:1])
        mask = torch.cat((pre_mask, mask), dim=0)
        
        preds = []
        for i in range(len(lr_pred)):
            sparse_inputs, coords = self.generate_sparse_inputs(x_hr[i:i+1], lr_pred[i: i+1], mask[i: i + 1])
            if coords.shape[0] > 0:
                pred = self.shm(sparse_inputs, lr_pred, coords, 1, mask.size()[2:], ctx=ctx[i: i + 1])
            else:
                # import pdb; pdb.set_trace()
                pred = [lr_pred[i:i+1]]
            preds.append(pred[-1])
        preds = torch.cat(preds, dim=0)
        
        last_pred = None
        all_hr_preds = []
        for i in range(len(lr_pred)):
            if last_pred is not None:
                last_pred = preds[i:i+1] * mask[i:i+1] + lr_pred[i:i+1] * (1-mask[i:i+1]) * (1-shared[i-1: i]) + last_pred * (1-mask[i:i+1]) * shared[i-1:i]
            else:
                last_pred = preds[i:i+1] * mask[i:i+1] + lr_pred[i:i+1] * (1-mask[i:i+1])
            all_hr_preds.append(last_pred)
        all_hr_preds = torch.cat(all_hr_preds, dim=0)

        all_hr_preds = all_hr_preds.view(bs, n_f, -1, *all_hr_preds.shape[-2:])
        return {'refined_masks': all_hr_preds}

    def forward(self, input_dict):
        # xlr = input_dict['lr_image'] # rescale to 256
        # xhr = input_dict['hr_image'] # 512

        # Resize image, mask, alpha, transition to 256
        # lr_pred, ctx = self.lpn(xlr)

        lr_inp = self.gen_lr_batch(input_dict, scale=0.5)
        with torch.no_grad():
            lr_out = self.mgm(lr_inp, return_ctx=True)

        lr_pred = lr_out['refined_masks'].clone().detach()
        ctx = lr_out['ctx'].clone().detach()

        # reshape to B, C, H, W before passing to SHM
        xhr = input_dict['image']
        b, n_f, _, h, w = xhr.shape
        xhr = xhr.view(b*n_f, -1, h, w)
        lr_pred = lr_pred.view(b*n_f, -1, *lr_pred.shape[-2:])

        # lr_pred = upas(lr_pred, xhr)
        lr_pred = F.interpolate(lr_pred, scale_factor=2.0, mode='bilinear', align_corners=False)
        lr_pred = lr_pred[:, :, :h, :w]
        
        if not self.training:
            return self.forward_inference(lr_pred, xhr, ctx, b, n_f)
        
        if 'transition' in input_dict:
            mask = input_dict['transition']
            mask = mask.view(b * n_f, -1, h, w)
        else:
            mask = self.dilate(lr_pred)

        n_pixel = mask.sum().item()
        

        # if n_pixel > 1700000:
        #     final_mask = lr_pred.reshape(b, n_f, -1, h, w)
        #     return {'refined_masks': final_mask}, loss_dict
        
        sparse_inputs, coords = self.generate_sparse_inputs(xhr, lr_pred, mask=mask)
        logging.debug("num pixels: {}".format(coords.shape))
        pred_list = self.shm(sparse_inputs, lr_pred, coords, xhr.size(0), mask.size()[2:], ctx=ctx)
        final_mask = pred_list[-1]
        final_mask = final_mask.reshape(b, n_f, -1, h, w)
        final_mask = final_mask * mask + lr_pred * (1-mask)
        output = {'refined_masks': final_mask}
        if self.training:
            # Compute loss
            loss_dict = self.compute_loss(pred_list, lr_pred, input_dict['alpha'], input_dict.get('fg', None), input_dict.get('bg', None), mask)
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
        if logit.numel() != target.numel():
            logit = upas(logit, target)
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
            
    def compute_loss(self, pred_list, lr_pred, alphas, fg, bg, mask):
        '''
        pred_list: list of predictions
        '''
        loss_dict = {}
        total_loss = 0
        alphas = alphas.view(-1, 1, *alphas.shape[-2:])
        if fg is not None:
            fg = fg.view(-1, 3, *fg.shape[-2:])
        if bg is not None:
            bg = bg.view(-1, 3, *bg.shape[-2:])
        if mask is not None:
            mask = mask.view(-1, 1, *mask.shape[-2:])

        pred_list = [upas(pred, alphas) for pred in pred_list]

        for i in range(len(pred_list)):
            pred_list[i] = pred_list[i] * mask + lr_pred * (1-mask)

        # Reg loss
        if self.loss_alpha_w > 0:
            loss_rec = 0
            weight = 2.0
            for pred in pred_list[::-1]:
                loss_rec += weight * self.regression_loss(pred, alphas, loss_type='l1', weight=mask)
                weight = weight / 2.0
            loss_dict['loss_rec'] = loss_rec
            total_loss += loss_dict['loss_rec'] * self.loss_alpha_w

        # Comp loss
        if self.loss_comp_w > 0 and fg is not None and bg is not None:
            loss = 0
            weight = 2.0
            for pred in pred_list[::-1]:
                loss += weight * loss_comp(pred, alphas, fg, bg, mask)
                weight = weight / 2.0
            loss_dict['loss_comp'] = loss
            total_loss += loss_dict['loss_comp'] * self.loss_comp_w

        # Lap loss
        if self.loss_alpha_lap_w > 0:
            loss = 0
            weight = 2.0
            for pred in pred_list[::-1]:
                loss += weight * self.lap_loss(pred, alphas, mask)
                weight = weight / 2.0
            loss_dict['loss_lap'] = loss
            total_loss += loss_dict['loss_lap'] * self.loss_alpha_lap_w

        loss_dict['total'] = total_loss
        return loss_dict

    def generate_sparsity_map(self, lr_pred, curr_img, last_img):
        mask_s = self.dilate(lr_pred)
        if last_img is not None:
            diff = (curr_img - last_img).abs().mean(dim=1, keepdim=True)
            shared = torch.logical_and(
                F.conv2d(diff, torch.ones(1,1,9,9,device=diff.device), padding=4) < 0.05,
                F.conv2d(diff, torch.ones(1,1,1,1,device=diff.device), padding=0) < 0.001,
            ).float()
            mask_t = self.dilate_op(1 - shared)
            mask = mask_s * mask_t
            mask = self.dilate_op(mask)
        else:
            shared = torch.zeros_like(mask_s)
            mask_t = torch.ones_like(mask_s)
            mask = mask_s * mask_t
        return mask, mask_s, mask_t, shared

    def inference(self, hr_img, lr_img=None, last_img=None, last_pred=None):
        # TODO: need to fix
        h, w = hr_img.shape[-2:]

        if lr_img is None:
            nh = 512. / min(h,w) * h
            nh = math.ceil(nh / 32) * 32
            nw = 512. / min(h,w) * w
            nw = math.ceil(nw / 32) * 32
            lr_img = F.interpolate(hr_img, (int(nh), int(nw)), mode="bilinear")

        lr_pred, ctx = self.lpn(lr_img)
        lr_pred_us = upas(lr_pred, hr_img)
        mask, mask_s, mask_t, shared = self.generate_sparsity_map(lr_pred_us, hr_img, last_img)
        n_pixel = mask.sum().item()

        if n_pixel <= self.max_n_pixel:
            sparse_inputs, coords = self.generate_sparse_inputs(hr_img, lr_pred_us, mask)
            preds = self.shm(sparse_inputs, lr_pred_us, coords, hr_img.size(0), mask.size()[2:], ctx=ctx)
            hr_pred_sp = preds[-1]
            if last_pred is not None:
                hr_pred = hr_pred_sp * mask + lr_pred_us * (1-mask) * (1-shared) + last_pred * (1-mask) * shared
            else:
                hr_pred = hr_pred_sp * mask + lr_pred_us * (1-mask)
        else:
            print("Rescaling is used.")
            scale = math.sqrt(self.max_n_pixel / float(n_pixel))
            nh = int(scale * h)
            nw = int(scale * w)
            nh = math.ceil(nh / 8) * 8
            nw = math.ceil(nw / 8) * 8

            hr_img_ds = F.interpolate(hr_img, (nh, nw), mode="bilinear")
            lr_pred_us = upas(lr_pred, hr_img_ds)
            mask_s = self.dilate(lr_pred_us)

            sparse_inputs, coords = self.generate_sparse_inputs(hr_img_ds, lr_pred_us, mask_s)
            preds = self.shm(sparse_inputs, lr_pred_us, coords, hr_img_ds.size(0), mask_s.size()[2:], ctx=ctx)
            hr_pred_sp = preds[-1]
            hr_pred = hr_pred_sp * mask_s + lr_pred_us * (1-mask_s)
        return hr_pred