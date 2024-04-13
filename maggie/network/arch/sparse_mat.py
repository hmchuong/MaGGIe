import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from ..encoder import *
from ..decoder import *
from ..loss import LapLoss, loss_comp


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
        x1, y1, x2, y2 = pos[i]
        x1 = int(x1.item() * w)
        y1 = int(y1.item() * h)
        x2 = int(x2.item() * w)
        y2 = int(y2.item() * h)
        patch = tensor[i:i+1, :, y1:y2, x1:x2].contiguous()
        patch = F.interpolate(patch, (size[0], size[1]), align_corners=False if mode=='bilinear' else None, mode=mode)
        patchs.append(patch)
    return torch.cat(patchs, dim=0)

class SparseMat(nn.Module, PyTorchModelHubMixin):
    def __init__(self, cfg):
        super(SparseMat, self).__init__()
        self.cfg = cfg
        self.lpn = eval(cfg.encoder)(**cfg.encoder_args) #MGM(backbone, decoder, cfg)
        self.shm = eval(cfg.decoder)(**cfg.decoder_args) #SHM(inc=4)
        self.lr_scale = cfg.shm.lr_scale
        self.stride = cfg.shm.dilation_kernel
        self.dilate_op = nn.MaxPool2d(self.stride, stride=1, padding=self.stride//2)
        self.max_n_pixel = cfg.shm.max_n_pixel

        self.loss_alpha_w = cfg.loss_alpha_w
        self.loss_alpha_lap_w = cfg.loss_alpha_lap_w
        self.lap_loss = LapLoss()

    @torch.no_grad()
    def generate_sparse_inputs(self, img, lr_pred, mask):
        lr_pred = (lr_pred - 0.5) / 0.5
        x = torch.cat((img, lr_pred), dim=1)
        indices = torch.where(mask.squeeze(1)>0)

        # print("num pixels: {}".format(len(indices[0])))
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
        # import pdb; pdb.set_trace()
        lr_batch['image'] = reshape5D(batch['image'], scale_factor=scale, multiply_by=64)
        mask_scale = scale / (batch['mask'].shape[-1] / batch['image'].shape[-1])
        lr_batch['mask'] = reshape5D(batch['mask'], scale_factor=mask_scale, multiply_by=64)
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

    def forward(self, input_dict, **kwargs):


        lr_inp = self.gen_lr_batch(input_dict, scale=0.5)
        xlr = torch.cat([lr_inp['image'], lr_inp['mask']], dim=2).flatten(0, 1)
        lr_pred, ctx = self.lpn(xlr)

        # reshape to B, C, H, W before passing to SHM
        xhr = input_dict['image']
        b, n_f, _, h, w = xhr.shape
        xhr = xhr.view(b*n_f, -1, h, w)
        lr_pred = lr_pred.view(b*n_f, -1, *lr_pred.shape[-2:])

        lr_pred = F.interpolate(lr_pred, scale_factor=2.0, mode='bilinear', align_corners=False)
        lr_pred = lr_pred[:, :, :h, :w]
        
        if not self.training:
            return self.forward_inference(lr_pred, xhr, ctx, b, n_f)
        
        mask = self.dilate(lr_pred)
        
        sparse_inputs, coords = self.generate_sparse_inputs(xhr, lr_pred, mask=mask)
        pred_list = self.shm(sparse_inputs, lr_pred, coords, xhr.size(0), mask.size()[2:], ctx=ctx)
        final_mask = pred_list[-1]
        final_mask = final_mask.reshape(b, n_f, -1, h, w)
        mask = mask.reshape(b, n_f, -1, h, w)
        lr_pred = lr_pred.reshape(b, n_f, -1, h, w)
        
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
        # import pdb; pdb.set_trace()
        mask = mask.view_as(alphas)
        lr_pred = lr_pred.view_as(alphas)
        for i in range(len(pred_list)):
            pred_list[i] = pred_list[i] * mask + lr_pred * (1-mask)

        # Reg loss
        if self.loss_alpha_w > 0:
            loss_rec = 0
            weight = 2.0
            for pred in pred_list[::-1]:
                # import pdb; pdb.set_trace()
                loss_rec += weight * self.regression_loss(pred, alphas, loss_type='l1', weight=None)
                weight = weight / 2.0
            loss_dict['loss_rec'] = loss_rec
            total_loss += loss_dict['loss_rec'] * self.loss_alpha_w

        # Lap loss
        if self.loss_alpha_lap_w > 0:
            loss = 0
            weight = 2.0
            for pred in pred_list[::-1]:
                loss += weight * self.lap_loss(pred, alphas, None)
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

class SparseMat_SingInst(SparseMat):
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