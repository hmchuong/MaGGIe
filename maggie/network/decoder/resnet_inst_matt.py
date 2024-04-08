import torch
import numpy as np
import cv2
from torch import nn
from torch.nn import functional as F

from .resnet import BasicBlock
from ..module import SpectralNorm, conv1x1, InstanceMatteDecoder
from ...utils.utils import compute_unknown, resizeAnyShape

class ResShortCut_InstMatt_Dec(nn.Module):
    def __init__(self, block, layers, 
                 norm_layer=None, large_kernel=False, late_downsample=False, 
                 atten_stride=1, atten_dim=128, atten_block=2, atten_head=1, final_channel=64,  max_inst=10, use_id_pe=True,
                 warmup_mask_atten_iter=4000, warmup_detail_iter=3000, use_query_temp=False, use_detail_temp=False, detail_mask_dropout=0.2, **kwargs):
        super(ResShortCut_InstMatt_Dec, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.large_kernel = large_kernel
        self.kernel_size = 5 if self.large_kernel else 3
        self.use_query_temp = use_query_temp
        self.use_detail_temp = use_detail_temp
        self.max_inst = max_inst

        self.inplanes = 512 if layers[0] > 0 else 256
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32
        self.warmup_mask_atten_iter = warmup_mask_atten_iter
        self.warmup_detail_iter = warmup_detail_iter

        self.conv1 = SpectralNorm(nn.ConvTranspose2d(self.midplanes, 32, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn1 = norm_layer(32)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.tanh = nn.Tanh()
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.midplanes, layers[3], stride=2)
        
        ## 1/8 scale
        self.refine_OS8 = InstanceMatteDecoder(
            input_dim=128,
            atten_stride=atten_stride,
            attention_dim=atten_dim,
            n_block=atten_block,
            n_head=atten_head,
            output_dim=final_channel,
            max_inst=max_inst,
            return_feat=True,
            use_temp_pe=False,
            use_id_pe=use_id_pe
        )

        ## OS4
        self.refine_OS4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(32, max_inst, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),)
        
        ## OS1
        self.refine_OS1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(32, max_inst, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2),
        )
        
        for module in [self.conv1, self.refine_OS1, self.refine_OS4]:
            if not isinstance(module, nn.Sequential):
                module = [self.conv1]
            for m in module:
                if isinstance(m, nn.Conv2d):
                    if hasattr(m, "weight_bar"):
                        nn.init.xavier_uniform_(m.weight_bar)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                else:
                    for p in m.parameters():
                        if p.dim() > 1:
                            nn.init.xavier_uniform_(p)


    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                SpectralNorm(conv1x1(self.inplanes, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, upsample, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def fuse(self, pred):
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        ### Progressive Refinement Module in MGMatting Paper
        alpha_pred = alpha_pred_os8.clone().detach()
        
        weight_os4 = compute_unknown(alpha_pred, k_size=30, train_mode=self.training)
        weight_os4 = weight_os4.type(alpha_pred.dtype)
        alpha_pred_os4 = alpha_pred_os4.type(alpha_pred.dtype)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4> 0]
        weight_os1 = compute_unknown(alpha_pred, k_size=15, train_mode=self.training)
        weight_os1 = weight_os1.type(alpha_pred.dtype)
        alpha_pred_os1 = alpha_pred_os1.type(alpha_pred.dtype)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1 > 0]

        return alpha_pred, weight_os4, weight_os1

    def forward(self, x, mid_fea, b, n_f, n_i, masks, iter, gt_alphas, **kwargs):
        '''
        masks: [b * n_f * n_i, 1, H, W]
        '''

        # Reshape masks
        masks = masks.reshape(b, n_f, n_i, masks.shape[2], masks.shape[3])
        valid_masks = masks.flatten(0,1).sum((2, 3), keepdim=True) > 0
        gt_masks = None
        if self.training:
            # In training, use gt_alphas to supervise the attention
            gt_masks = (gt_alphas > 0).reshape(b, n_f, n_i, gt_alphas.shape[2], gt_alphas.shape[3])

            if gt_masks.shape[-1] != masks.shape[-1]:
                gt_masks = resizeAnyShape(gt_masks, scale_factor=masks.shape[-1] * 1.0/ gt_masks.shape[-1], use_max_pool=True)

        # OS32 -> OS 8
        
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        
        image = mid_fea['image']
        x = self.layer1(x) + fea5

        x = self.layer2(x) + fea4

        h, w = image.shape[-2:]

        # Predict OS8
        # use mask attention during warmup of training
        use_mask_atten = iter < self.warmup_mask_atten_iter and self.training
        x_os8, x, _, loss_max_atten, _ = self.refine_OS8(x, masks, use_mask_atten=use_mask_atten, gt_mask=gt_masks)
        x_os8 = F.interpolate(x_os8, size=(h, w), mode='bilinear', align_corners=False)

        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0
        if self.training:
            x_os8 = x_os8 * valid_masks
        else:
            x_os8 = x_os8[:, :n_i]

        unknown_os8 = compute_unknown(x_os8, k_size=30)

        x = self.layer3(x) + fea3
        x_os4 = self.refine_OS4(x)

        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x_os1 = self.refine_OS1(x)

        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        
        x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0

        if not self.training:
            x_os4 = x_os4[:, :n_i]
            x_os1 = x_os1[:, :n_i]

        ret = {}
        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8

        alpha_pred, weight_os4, weight_os1 = self.fuse(ret)

        ret['refined_masks'] = alpha_pred
        
        ret['weight_os4'] = weight_os4
        ret['weight_os1'] = weight_os1
        ret['detail_mask'] = unknown_os8
        
        # ret['diff_pred'] = diff_pred
        if self.training and iter >= self.warmup_mask_atten_iter:
            ret['loss_max_atten'] = loss_max_atten
        return ret

def res_shortcut_inst_matt_22(**kwargs):
    return ResShortCut_InstMatt_Dec(BasicBlock, [2, 3, 3, 2], **kwargs)