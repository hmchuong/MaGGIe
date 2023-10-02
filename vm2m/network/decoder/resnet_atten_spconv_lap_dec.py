from functools import partial
import logging
import torch
import random
import cv2
import numpy as np
from torch import nn
from torch.nn import functional as F
import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp
from vm2m.network.ops import SpectralNorm
from vm2m.network.module.base import conv1x1
from vm2m.network.module.mask_matte_embed_atten import MaskMatteEmbAttenHead
from vm2m.network.module.stm_2 import STM
from vm2m.network.module.detail_aggregation import DetailAggregation
from vm2m.utils.utils import compute_unknown
from .resnet_dec import BasicBlock

class ResShortCut_AttenSpconv_Lap_Dec(nn.Module):
    def __init__(self, block, layers, norm_layer=None, large_kernel=False, 
                 late_downsample=False, final_channel=32,
                 atten_dim=128, atten_block=2, 
                 atten_head=1, atten_stride=1, max_inst=10, warmup_mask_atten_iter=4000, use_id_pe=True, use_query_temp=False, use_detail_temp=False, detail_mask_dropout=0.2, warmup_detail_iter=3000, **kwargs):
        super(ResShortCut_AttenSpconv_Lap_Dec, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.large_kernel = large_kernel
        self.kernel_size = 5 if self.large_kernel else 3
        self.use_query_temp = use_query_temp
        self.use_detail_temp = use_detail_temp

        self.inplanes = 512 if layers[0] > 0 else 256
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32
        self.warmup_mask_atten_iter = warmup_mask_atten_iter
        self.warmup_detail_iter = warmup_detail_iter

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.tanh = nn.Tanh()
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        # Sparse Conv from OS4 to OS2
        self.layer4 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(64, 64, kernel_size=3, bias=False, indice_key="subm4.0"),
            nn.BatchNorm1d(64),
            self.leaky_relu,
            spconv.SubMConv2d(64, 32, kernel_size=3, padding=1, bias=False, indice_key="subm2.2"),
        )

        # Sparse Conv from OS2 to OS1
        self.layer5 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(32, 32, kernel_size=3, bias=False, indice_key="subm2.0"),
            nn.BatchNorm1d(32),
            self.leaky_relu,
            spconv.SubMConv2d(32, 32, kernel_size=3, padding=1, bias=False, indice_key="subm1.2"),
        )
        
        ## 1/8 scale
        self.refine_OS8 = MaskMatteEmbAttenHead(
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
        relu_layer = nn.ReLU(inplace=True)
        # Image low-level feature extractor
        self.low_os1_module = spconv.SparseSequential(
            spconv.SubMConv2d(4, 32, kernel_size=3, padding=1, bias=False, indice_key="subm1.0"),
            relu_layer,
            nn.BatchNorm1d(32),
            spconv.SubMConv2d(32, 32, kernel_size=3, padding=1, bias=False, indice_key="subm1.1"),
            relu_layer,
            nn.BatchNorm1d(32)
        )

        self.low_os2_module = spconv.SparseSequential(
            spconv.SparseConv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False, indice_key="subm2.0"),
            relu_layer,
            nn.BatchNorm1d(32),
            spconv.SubMConv2d(32, 32, kernel_size=3, padding=1, bias=False, indice_key="subm2.1"),
            relu_layer,
            nn.BatchNorm1d(32)
        )

        self.low_os4_module = spconv.SparseSequential(
            spconv.SparseConv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False, indice_key="subm4.0"),
            relu_layer,
            nn.BatchNorm1d(64),
            spconv.SubMConv2d(64, 64, kernel_size=3, padding=1, bias=False, indice_key="subm4.1"),
            relu_layer,
            nn.BatchNorm1d(64)
        )

        self.refine_OS4 = spconv.SparseSequential(
            spconv.SubMConv2d(64, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            nn.BatchNorm1d(32),
            self.leaky_relu,
            spconv.SubMConv2d(32, 1, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        ) 

        self.refine_OS1 = spconv.SparseSequential(
            spconv.SubMConv2d(32, 32, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
            nn.BatchNorm1d(32),
            self.leaky_relu,
            spconv.SubMConv2d(32, 1, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        )

        # self.mask_dropout_p = detail_mask_dropout
        self.fea_dropout = nn.Dropout2d(detail_mask_dropout)

    def convert_syn_bn(self):
        self.layer1 = nn.SyncBatchNorm.convert_sync_batchnorm(self.layer1)
        self.layer2 = nn.SyncBatchNorm.convert_sync_batchnorm(self.layer2)
        self.layer3 = nn.SyncBatchNorm.convert_sync_batchnorm(self.layer3)
        self.refine_OS8 = nn.SyncBatchNorm.convert_sync_batchnorm(self.refine_OS8)


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
    
    def aggregate_detail_mem(self, x, mem_details):
        return x

    def predict_details(self, x, image, roi_masks, masks, mem_details=None):
        '''
        x: [b, 64, h/4, w/4], os4 semantic features
        image: [b, 3, H, W], original image
        masks: [b, n_i, H, W], dilated guided masks from OS8 prediction
        mem_details: SparseConvTensor, memory details
        '''
        # Stack masks and images
        b, n_i, H, W = masks.shape
        masks = masks.reshape(b * n_i, 1, H, W)
        roi_masks = roi_masks.reshape(b * n_i, 1, H, W)
        image = image.unsqueeze(1).repeat(1, n_i, 1, 1, 1).reshape(b * n_i, image.shape[1], H, W)
            

        inp = torch.cat([image, masks], dim=1)

        # Prepare sparse tensor
        coords = torch.where(roi_masks.squeeze(1) > 0)
        # if self.training and len(coords[0]) > 1600000:
        #     ids = torch.randperm(len(coords[0]))[:1600000]
        #     coords = [i[ids] for i in coords]
        
        inp = inp.permute(0, 2, 3, 1).contiguous()
        inp = inp[coords]
        coords = torch.stack(coords, dim=1)
        inp = spconv.SparseConvTensor(inp, coords.int(), (H, W), b * n_i)

        # inp -> OS 1 --> OS 2 --> OS 4
        fea1 = self.low_os1_module(inp)
        fea2 = self.low_os2_module(fea1)
        fea3 = self.low_os4_module(fea2)

        # Combine x with fea3
        # Prepare sparse tensor of x
        coords = fea3.indices.clone()
        coords[:, 0] = torch.div(coords[:, 0], n_i, rounding_mode='floor')
        coords = coords.long()
        x = self.fea_dropout(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x[coords[:, 0], coords[:, 1], coords[:, 2]]
        x = spconv.SparseConvTensor(x, fea3.indices, (H // 4, W // 4), b * n_i, indice_dict=fea3.indice_dict)
        x = Fsp.sparse_add(x, fea3)

        # import pdb; pdb.set_trace()
        if mem_details is not None and self.use_detail_temp:
            x = self.aggregate_detail_mem(x, mem_details)
        mem_details = x

        # Predict OS 4
        x_os4 = self.refine_OS4(x)
        x_os4_out = x_os4.dense()
        x_os4_out = x_os4_out - 99
        coords = x_os4.indices.long()
        x_os4_out[coords[:, 0], :, coords[:, 1], coords[:, 2]] += 99

        # Combine with fea2
        # - Upsample to OS 2
        x = self.layer4(x)
        x = Fsp.sparse_add(x, fea2)

        # Combine with fea1
        # - Upsample to OS 1
        x = self.layer5(x)
        x = Fsp.sparse_add(x, fea1)

        # Predict OS 1
        x_os1 = self.refine_OS1(x)
        x_os1_out = x_os1.dense()
        x_os1_out = x_os1_out - 99
        coords = x_os1.indices.long()
        x_os1_out[coords[:, 0], :, coords[:, 1], coords[:, 2]] += 99

        return x_os4_out, x_os1_out, mem_details
        # dilated_m = F.max_pool2d(uncertain.float(), kernel_size=k_size, stride=1, padding=k_size // 2)
        # dilated_m = dilated_m[:,:, :h, :w]
        # return dilated_m

    def aggregate_mem(self, x, mem_feat):
        '''
        Update current feat with mem_feat
        '''
        return x

    def update_mem(self, mem_feat, refined_masks):
        '''
        Update memory feature with refined masks
        '''
        return mem_feat

    def update_detail_mem(self, mem_details, refined_masks):
        return mem_details

    def fuse_pred(self, pred, detail_mask):
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        alpha_pred = alpha_pred_os8.clone()
        
        # Upsample and replace by OS4
        alpha_pred = F.interpolate(alpha_pred, size=alpha_pred_os4.shape[-2:], mode='bilinear', align_corners=False)
        unknown_os4 = compute_unknown(alpha_pred, k_size=8) * detail_mask[:, :, ::4, ::4]
        alpha_pred = alpha_pred_os4 * unknown_os4 + alpha_pred * (1 - unknown_os4)

        # cv2.imwrite('alpha_pred_os4.png', alpha_pred[0,0].cpu().numpy() * 255)
        # cv2.imwrite('alpha_pred_os8.png', alpha_pred_os8[0,0].cpu().numpy() * 255)
        # cv2.imwrite('alpha_unk_os4.png', unknown_os4[0,0].cpu().numpy() * 255)
        # cv2.imwrite('alpha_unk_os1.png', unknown_os1[0,0].cpu().numpy() * 255)
        # import pdb; pdb.set_trace()

        # Upsample and replace by OS1
        alpha_pred = F.interpolate(alpha_pred, size=alpha_pred_os1.shape[-2:], mode='bilinear', align_corners=False)
        unknown_os1 = compute_unknown(alpha_pred, k_size=15) * detail_mask
        alpha_pred = alpha_pred_os1 * unknown_os1 + alpha_pred * (1 - unknown_os1)

        return unknown_os1, unknown_os4, alpha_pred
    
    def forward(self, x, mid_fea, b, n_f, n_i, masks, iter, gt_alphas, mem_feat=None, mem_query=None, mem_details=None, lap_image=None, **kwargs):
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

        # OS32 -> OS 8
        ret = {}
        fea4, fea5 = mid_fea['shortcut']
        
        image = mid_fea['image']
        x = self.layer1(x) + fea5
        
        # Update with memory
        x = self.aggregate_mem(x, mem_feat)
        mem_feat = x
        x = self.layer2(x) + fea4

        # Predict OS8
        # use mask attention during warmup of training
        use_mask_atten = iter < self.warmup_mask_atten_iter and self.training
        # import pdb; pdb.set_trace()
        x_os8, x, queries, loss_max_atten, loss_min_atten = self.refine_OS8(x, masks, 
                                                                            prev_tokens=mem_query if self.use_query_temp else None, 
                                                                            use_mask_atten=use_mask_atten, gt_mask=gt_masks)
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0
        up_x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)

        if self.training:
            up_x_os8 = up_x_os8 * valid_masks
        else:
            x_os8 = x_os8[:, :n_i]
            up_x_os8 = up_x_os8[:, :n_i]

        x = self.layer3(x)

        # Warm-up - Using gt_alphas instead of x_os8 for later steps
        guided_mask_os8 = up_x_os8
        if self.training and (iter < self.warmup_detail_iter or up_x_os8.sum() == 0 or (iter < self.warmup_detail_iter * 3 and random.random() < 0.5)):
            guided_mask_os8 = gt_alphas.clone()

            # Downscale and upscale to simulate the missed information
            guided_mask_os8 = F.avg_pool2d(guided_mask_os8, kernel_size=8, stride=8)
            guided_mask_os8 = F.interpolate(guided_mask_os8, scale_factor=8.0, mode='bilinear', align_corners=False)

        # Compute RoI for next level
        unknown_os8 = compute_unknown(guided_mask_os8)

        if unknown_os8.sum() > 0 or self.training:
            # Combine with details memory
            # detail_input = torch.cat([image, lap_image], dim=1)
            detail_input = image
            x_os4, x_os1, mem_details = self.predict_details(x, detail_input, unknown_os8, guided_mask_os8, mem_details)
            x_os4 = x_os4.reshape(b * n_f, guided_mask_os8.shape[1], *x_os4.shape[-2:])
            x_os1 = x_os1.reshape(b * n_f, guided_mask_os8.shape[1], *x_os1.shape[-2:])
            x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
            x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        else:
            x_os4 = torch.zeros((b * n_f, x_os8.shape[1], image.shape[2]//4, image.shape[3]//4), device=x_os8.device)
            x_os1 = torch.zeros((b * n_f, x_os8.shape[1], image.shape[2], image.shape[3]), device=x_os8.device)
        
        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8

        weight_os1, weight_os4, alpha_pred = self.fuse_pred(ret, unknown_os8)
        ret['weight_os1'] = weight_os1
        ret['weight_os4'] = weight_os4
        ret['alpha'] = alpha_pred

        mem_feat = self.update_mem(mem_feat, alpha_pred)

        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
        x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        
        # Update mem_details
        mem_details = self.update_detail_mem(mem_details, alpha_pred)

        ret['detail_mask'] = unknown_os8
        if self.training and iter >= self.warmup_mask_atten_iter:
            ret['loss_max_atten'] = loss_max_atten
            ret['loss_min_atten'] = loss_min_atten
        ret['mem_queries'] = queries
        ret['mem_feat'] = mem_feat
        ret['mem_details'] = mem_details
        return ret

def res_shortcut_attention_spconv_lap_decoder_22(**kwargs):
    return ResShortCut_AttenSpconv_Lap_Dec(BasicBlock, [2, 3, 3, 2], **kwargs)