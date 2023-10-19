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
from vm2m.network.module.base import conv1x1, conv3x3
from vm2m.network.module.mask_matte_embed_atten import MaskMatteEmbAttenHead
from vm2m.network.module.instance_matte_head import InstanceMatteHead
from vm2m.network.module.temporal_nn import TemporalNN
from vm2m.network.module.ligru_conv import LiGRUConv
from vm2m.network.module.stm_2 import STM
from vm2m.network.module.conv_gru import ConvGRU
from vm2m.network.module.stm_window import WindowSTM
from vm2m.network.module.detail_aggregation import DetailAggregation
from vm2m.network.module.instance_query_atten import InstanceQueryAttention
from vm2m.utils.utils import compute_unknown
from .resnet_dec import BasicBlock

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

class ResShortCut_AttenSpconv_Dec(nn.Module):
    def __init__(self, block, layers, norm_layer=None, large_kernel=False, 
                 late_downsample=False, final_channel=32,
                 atten_dim=128, atten_block=2, 
                 atten_head=1, atten_stride=1, max_inst=10, warmup_mask_atten_iter=4000,
                  use_id_pe=True, use_query_temp=False, use_detail_temp=False, detail_mask_dropout=0.2, warmup_detail_iter=3000, **kwargs):
        super(ResShortCut_AttenSpconv_Dec, self).__init__()
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
            use_id_pe=use_id_pe,
            temporal_query=use_query_temp
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
    
    def aggregate_detail_mem(self, x, mem_details, n_i):
        return x, mem_details

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
        image = image.unsqueeze(1).repeat(1, n_i, 1, 1, 1).reshape(b * n_i, 3, H, W)
        
        # if self.training and self.mask_dropout_p > 0:
        #     nonzeros_ids = torch.nonzero(masks)
        #     if len(nonzeros_ids) > 0:
        #         mask_ids = torch.randperm(len(nonzeros_ids))[:int(len(nonzeros_ids) * self.mask_dropout_p)]
        #         selected_ids = nonzeros_ids[mask_ids]
        #         masks[selected_ids[:, 0], selected_ids[:, 1], selected_ids[:, 2], selected_ids[:, 3]] = 0
            

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

        # fea3, mem_details = self.aggregate_detail_mem(fea3, mem_details, n_i)

        # Combine x with fea3
        # Prepare sparse tensor of x
        coords = fea3.indices.clone()
        coords[:, 0] = torch.div(coords[:, 0], n_i, rounding_mode='floor')
        coords = coords.long()
        x = self.fea_dropout(x)
        # import pdb; pdb.set_trace()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x[coords[:, 0], coords[:, 1], coords[:, 2]]
        x = spconv.SparseConvTensor(x, fea3.indices, (H // 4, W // 4), b * n_i, indice_dict=fea3.indice_dict)
        x = Fsp.sparse_add(x, fea3)

        # import pdb; pdb.set_trace()
        # if mem_details is not None and self.use_detail_temp:
        x, mem_details = self.aggregate_detail_mem(x, mem_details, n_i)
        # mem_details = x

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

    # def compute_unknown(self, masks, k_size=30):
    #     h, w = masks.shape[-2:]
    #     uncertain = (masks > 1.0/255.0) & (masks < 254.0/255.0)
    #     dilated_m = F.max_pool2d(uncertain.float(), kernel_size=k_size, stride=1, padding=k_size // 2)
    #     dilated_m = dilated_m[:,:, :h, :w]
    #     return dilated_m

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

    def compute_diff(self, x, prev_feat):
        '''
        Compute difference between current and previous frame input
        '''
        return None

    def fushion(self, pred, detail_mask):
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        ### Progressive Refinement Module in MGMatting Paper
        alpha_pred = alpha_pred_os8.clone() #.detach()
        
        # weight_os4 = get_unknown_tensor_from_pred(alpha_pred, rand_width=30, train_mode=self.training)
        weight_os4 = compute_unknown(alpha_pred, k_size=30) * detail_mask
        weight_os4 = weight_os4.type(alpha_pred.dtype)
        alpha_pred_os4 = alpha_pred_os4.type(alpha_pred.dtype)
        weight_os4 = weight_os4 * (alpha_pred_os4 > 1.0/255).float()
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4> 0]
        # cv2.imwrite("test_fuse.png", alpha_pred[0,0].detach().cpu().numpy() * 255)
        # cv2.imwrite("test_fuse_wos4.png", weight_os4[0,0].detach().cpu().numpy() * 255)
        # cv2.imwrite("test_fuse_os8.png", alpha_pred_os8[0,0].detach().cpu().numpy() * 255)
        # cv2.imwrite("test_fuse_os4.png", alpha_pred_os4[0,0].detach().cpu().numpy() * 255)
        # import pdb; pdb.set_trace()

        # weight_os1 = get_unknown_tensor_from_pred(alpha_pred, rand_width=15, train_mode=self.training)
        weight_os1 = compute_unknown(alpha_pred, k_size=15) * detail_mask
        weight_os1 = weight_os1.type(alpha_pred.dtype)
        alpha_pred_os1 = alpha_pred_os1.type(alpha_pred.dtype)
        weight_os1 = weight_os1 * (alpha_pred_os1 > 1.0/255).float()
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1 > 0]

        return alpha_pred, weight_os4, weight_os1

    def forward(self, x, mid_fea, b, n_f, n_i, masks, iter, gt_alphas, mem_feat=None, mem_query=None, mem_details=None, **kwargs):
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
        # x = self.aggregate_mem(x, mem_feat)
        # mem_feat = x

        x = self.layer2(x) + fea4

        

        # Predict OS8
        # import pdb; pdb.set_trace()
        # use mask attention during warmup of training
        use_mask_atten = iter < self.warmup_mask_atten_iter and self.training
        # import pdb; pdb.set_trace()
        x_os8, x, queries, loss_max_atten, loss_min_atten = self.refine_OS8(x, masks, 
                                                                            prev_tokens=mem_query if self.use_query_temp else None, 
                                                                            use_mask_atten=use_mask_atten, gt_mask=gt_masks)

        # Compute differences between current and previous frame
        # diff_pred = self.compute_diff(x, mem_feat[0] if mem_feat is not None else None)
        # prev_pred = mem_feat[1] if mem_feat is not None else None
        mem_feat = x

        # If not training, compute new os8 based on diff_pred
        # if not self.training and prev_pred is not None and diff_pred is not None:
        #     x_os8 = prev_pred * (1 - diff_pred) + diff_pred * x_os8
        #     prev_pred = x_os8

        # import pdb; pdb.set_trace()
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0
        if self.training:
            x_os8 = x_os8 * valid_masks
        else:
            x_os8 = x_os8[:, :n_i]

        x = self.layer3(x)

        # Warm-up - Using gt_alphas instead of x_os8 for later steps
        guided_mask_os8 = x_os8
        if self.training and (iter < self.warmup_detail_iter or x_os8.sum() == 0 or (iter < self.warmup_detail_iter * 3 and random.random() < 0.5)):
            guided_mask_os8 = gt_alphas.clone()
            # if gt_alphas.max() == 0 and self.training:
            #     guided_mask_os8[:, :, 200: 250, 200: 250] = 1.0
            # print(guided_mask_os8.sum(), guided_mask_os8.max())

        

        # unknown_os8 = self.compute_unknown(guided_mask_os8)
        unknown_os8 = compute_unknown(guided_mask_os8, k_size=30, is_train=self.training)

        if unknown_os8.max() == 0 and self.training:
            unknown_os8[:, :, 200: 250, 200: 250] = 1.0

        if unknown_os8.sum() > 0 or self.training:
            # TODO: Combine with details memory
            # guided_mask_os8 = (guided_mask_os8 > 0).float()
            x_os4, x_os1, mem_details = self.predict_details(x, image, unknown_os8, guided_mask_os8, mem_details)
            x_os4 = x_os4.reshape(b * n_f, guided_mask_os8.shape[1], *x_os4.shape[-2:])
            x_os1 = x_os1.reshape(b * n_f, guided_mask_os8.shape[1], *x_os1.shape[-2:])

            x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
            x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
            x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        else:
            x_os4 = torch.zeros((b * n_f, x_os8.shape[1], image.shape[2], image.shape[3]), device=x_os8.device)
            x_os1 = torch.zeros_like(x_os4)
        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8

        # if not self.training:
        #     import pdb; pdb.set_trace()
        # Update mem_feat
        alpha_pred, _, _ = self.fushion(ret, unknown_os8)
        # mem_feat = self.update_mem(mem_feat, alpha_pred)
        # mem_feat = (mem_feat, prev_pred)

        # Update mem_details
        mem_details = self.update_detail_mem(mem_details, alpha_pred)

        ret['refined_masks'] = alpha_pred
        # ret['refined_masks'] = x_os8
        # ret['weight_os4'] = weight_os4
        # ret['weight_os1'] = weight_os1
        ret['detail_mask'] = unknown_os8
        # ret['diff_pred'] = diff_pred
        if self.training and iter >= self.warmup_mask_atten_iter:
            ret['loss_max_atten'] = loss_max_atten
            ret['loss_min_atten'] = loss_min_atten
        ret['mem_queries'] = queries
        ret['mem_feat'] = mem_feat
        ret['mem_details'] = mem_details
        return ret

class ResShortCut_AttenSpconv_Temp_Dec(ResShortCut_AttenSpconv_Dec):
    
    def __init__(self, stm_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.temp_module_os16 = STM(256, os=16, mask_channel=kwargs["embed_dim"], drop_out=stm_dropout)
        if self.use_detail_temp:
            self.temp_module_os4 = DetailAggregation(64)

    def convert_syn_bn(self):
        super().convert_syn_bn()
        self.temp_module_os16 = nn.SyncBatchNorm.convert_sync_batchnorm(self.temp_module_os16)

    def aggregate_mem(self, x, mem_feat):
        '''
        Update current feat with mem_feat

        x: b, c, h, w
        mem_feat: b, t, c, h, w
        '''
        if mem_feat is not None:
            x = self.temp_module_os16(x, mem_feat)
        return x

    def update_mem(self, mem_feat, refined_masks):
        '''
        Update memory feature with refined masks
        '''
        mem_feat = self.temp_module_os16.generate_mem(mem_feat, refined_masks)
        return mem_feat

    def aggregate_detail_mem(self, x, mem_details):
        if mem_details is not None:
            x = self.temp_module_os4(x, mem_details)
        return x, mem_details
    
    def update_detail_mem(self, mem_details, refined_masks):
        if mem_details is not None and self.use_detail_temp:
            mem_details = self.temp_module_os4.generate_mem(mem_details, refined_masks)
        return mem_details
    
    def forward(self, x, mid_fea, b, n_f, masks, mem_feat=[], mem_query=None, mem_details=None, gt_alphas=None, **kwargs):
        
        # Reshape inputs
        x = x.reshape(b, n_f, *x.shape[1:])
        if gt_alphas is not None:
            gt_alphas = gt_alphas.reshape(b, n_f, *gt_alphas.shape[1:])
        image = mid_fea['image']
        image = image.reshape(b, n_f, *image.shape[1:])
        masks = masks.reshape(b, n_f, *masks.shape[1:])
        fea4 = mid_fea['shortcut'][0]
        fea4 = fea4.reshape(b, n_f, *fea4.shape[1:])
        fea5 = mid_fea['shortcut'][1]
        fea5 = fea5.reshape(b, n_f, *fea5.shape[1:])
        
        final_results = {}
        need_construct_mem = len(mem_feat) == 0
        # import pdb; pdb.set_trace()
        n_mem = 5
        for i in range(n_f):
            mid_fea = {
                'image': image[:, i],
                'shortcut': [fea4[:, i], fea5[:, i]]
            }
            
            # Construct memory
            input_mem_feat = mem_feat
            if need_construct_mem:
                if len(mem_feat) > 0:
                    input_mem_feat = [mem_feat[-1]]
                if len(mem_feat) > 1:
                    input_mem_feat = [mem_feat[max(0, len(mem_feat) - 1 - n_mem)].detach()] + input_mem_feat
            if len(input_mem_feat) > 0:
                input_mem_feat = torch.cat(input_mem_feat, dim=1)
            else:
                input_mem_feat = None
            # import pdb; pdb.set_trace()
            ret = super().forward(x=x[:, i], mid_fea=mid_fea, b=b, n_f=1, masks=masks[:, i], mem_feat=input_mem_feat, mem_query=mem_query, mem_details=mem_details, gt_alphas=gt_alphas[:,i] if gt_alphas is not None else gt_alphas, **kwargs)
            mem_query = ret['mem_queries'].detach()
            mem_details = ret['mem_details']
            mem_feat += [ret['mem_feat'].unsqueeze(1)]

            for k in ret:
                if k not in final_results:
                    final_results[k] = []
                final_results[k] += [ret[k]]
        for k, v in final_results.items():
            if k in ['alpha_os1', 'alpha_os4', 'alpha_os8', 'weight_os4', 'weight_os1', 'refined_masks']:
                final_results[k] = torch.stack(v, dim=1).flatten(0, 1)
            elif k in ['mem_feat', 'mem_details', 'mem_queries']:
                final_results[k] = v[-1]
            elif self.training:
                final_results[k] = torch.stack(v).mean()
        return final_results

class ResShortCut_AttenSpconv_QueryTemp_Dec(ResShortCut_AttenSpconv_Dec):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.use_detail_temp:
            self.temp_module_os4 = InstanceQueryAttention(64, max_inst=self.max_inst)
    
    def aggregate_detail_mem(self, x, mem_details, n_i):
        '''
        Update current feat with mem_feat
        '''
        if self.use_detail_temp:
            return self.temp_module_os4(x, n_i, mem_details)
        return x, mem_details

    # def forward(self, x, mid_fea, b, n_f, masks, mem_query=None, mem_details=None, gt_alphas=None, **kwargs):
        
    #     # Reshape inputs
    #     x = x.reshape(b, n_f, *x.shape[1:])
    #     if gt_alphas is not None:
    #         gt_alphas = gt_alphas.reshape(b, n_f, *gt_alphas.shape[1:])
    #     image = mid_fea['image']
    #     image = image.reshape(b, n_f, *image.shape[1:])
    #     masks = masks.reshape(b, n_f, *masks.shape[1:])
    #     fea4 = mid_fea['shortcut'][0]
    #     fea4 = fea4.reshape(b, n_f, *fea4.shape[1:])
    #     fea5 = mid_fea['shortcut'][1]
    #     fea5 = fea5.reshape(b, n_f, *fea5.shape[1:])
        
    #     final_results = {}
    #     # import pdb; pdb.set_trace()
    #     n_mem = 5
    #     for i in range(n_f):
    #         mid_fea = {
    #             'image': image[:, i],
    #             'shortcut': [fea4[:, i], fea5[:, i]]
    #         }
    #         ret = super().forward(x=x[:, i], mid_fea=mid_fea, b=b, n_f=1, masks=masks[:, i], mem_query=mem_query, mem_details=mem_details, gt_alphas=gt_alphas[:,i] if gt_alphas is not None else gt_alphas, **kwargs)
    #         mem_query = ret['mem_queries']
    #         mem_details = ret['mem_details']

    #         for k in ret:
    #             if k not in final_results:
    #                 final_results[k] = []
    #             final_results[k] += [ret[k]]
    #     for k, v in final_results.items():
    #         if k in ['alpha_os1', 'alpha_os4', 'alpha_os8', 'weight_os4', 'weight_os1', 'refined_masks']:
    #             final_results[k] = torch.stack(v, dim=1).flatten(0, 1)
    #         elif k in ['mem_feat', 'mem_details', 'mem_queries']:
    #             final_results[k] = v[-1]
    #         elif self.training:
    #             final_results[k] = torch.stack(v).mean()
    #     return final_results
        
    def forward(self, x, mid_fea, b, n_f, masks, mem_query=None, mem_details=None, gt_alphas=None, **kwargs):
        
        # Reshape inputs
        x = x.reshape(b, n_f, *x.shape[1:])
        if gt_alphas is not None:
            gt_alphas = gt_alphas.reshape(b, n_f, *gt_alphas.shape[1:])
        image = mid_fea['image']
        image = image.reshape(b, n_f, *image.shape[1:])
        masks = masks.reshape(b, n_f, *masks.shape[1:])
        fea4 = mid_fea['shortcut'][0]
        fea4 = fea4.reshape(b, n_f, *fea4.shape[1:])
        fea5 = mid_fea['shortcut'][1]
        fea5 = fea5.reshape(b, n_f, *fea5.shape[1:])
        
        final_results = {}
        final_results_notemp = {}
        n_mem = 5
        for i in range(n_f):
            mid_fea = {
                'image': image[:, i],
                'shortcut': [fea4[:, i], fea5[:, i]]
            }
            ret = super().forward(x=x[:, i], mid_fea=mid_fea, b=b, n_f=1, masks=masks[:, i], mem_query=mem_query, mem_details=mem_details, gt_alphas=gt_alphas[:,i] if gt_alphas is not None else gt_alphas, **kwargs)
            mem_query = ret['mem_queries']
            mem_details = ret['mem_details']

            for k in ret:
                if k not in final_results:
                    final_results[k] = []
                final_results[k] += [ret[k]]
        
        # Compute new temp loss to focus on wrong regions
        for k, v in final_results.items():
            if k in ['alpha_os1', 'alpha_os4', 'alpha_os8', 'weight_os4', 'weight_os1', 'refined_masks', 'detail_mask']:
                final_results[k] = torch.stack(v, dim=1).flatten(0, 1)
            elif k in ['mem_feat', 'mem_details', 'mem_queries']:
                final_results[k] = v[-1]
            elif self.training:
                final_results[k] = torch.stack(v).mean()
        return final_results

class ResShortCut_AttenSpconv_InconsistTemp_Dec(ResShortCut_AttenSpconv_Dec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Module to compute the difference between two inputs
        self.diff_module = nn.Sequential(
            SpectralNorm(conv1x1(256, 64)),
            self._norm_layer(64),
            nn.ReLU(inplace=True),
            SpectralNorm(conv3x3(64, 32)),
            self._norm_layer(32),
            nn.ReLU(inplace=True),
            conv3x3(32, 1)
        )

    def compute_diff(self, x, prev_feat):
        '''
        Compute difference between current and previous frame input
        '''
        if prev_feat is None:
            return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device) * 99
        inp = torch.cat([x, prev_feat], dim=1)
        return self.diff_module(inp)

    def forward(self, x, mid_fea, b, n_f, masks, mem_feat=None, mem_query=None, mem_details=None, gt_alphas=None, **kwargs):
        # Reshape inputs
        x = x.reshape(b, n_f, *x.shape[1:])
        if gt_alphas is not None:
            gt_alphas = gt_alphas.reshape(b, n_f, *gt_alphas.shape[1:])
        image = mid_fea['image']
        image = image.reshape(b, n_f, *image.shape[1:])
        masks = masks.reshape(b, n_f, *masks.shape[1:])
        fea4 = mid_fea['shortcut'][0]
        fea4 = fea4.reshape(b, n_f, *fea4.shape[1:])
        fea5 = mid_fea['shortcut'][1]
        fea5 = fea5.reshape(b, n_f, *fea5.shape[1:])
        
        final_results = {}
        final_results_notemp = {}
        n_mem = 5
        for i in range(n_f):
            mid_fea = {
                'image': image[:, i],
                'shortcut': [fea4[:, i], fea5[:, i]]
            }
            ret = super().forward(x=x[:, i], mid_fea=mid_fea, b=b, n_f=1, masks=masks[:, i], mem_feat=mem_feat, mem_query=mem_query, mem_details=mem_details, gt_alphas=gt_alphas[:,i] if gt_alphas is not None else gt_alphas, **kwargs)
            mem_query = ret['mem_queries']
            mem_details = ret['mem_details']
            mem_feat = ret['mem_feat']

            for k in ret:
                if k not in final_results:
                    final_results[k] = []
                final_results[k] += [ret[k]]
        
        # Compute new temp loss to focus on wrong regions
        for k, v in final_results.items():
            if k in ['alpha_os1', 'alpha_os4', 'alpha_os8', 'weight_os4', 'weight_os1', 'refined_masks', 'detail_mask', 'diff_pred']:
                final_results[k] = torch.stack(v, dim=1).flatten(0, 1)
            elif k in ['mem_feat', 'mem_details', 'mem_queries']:
                final_results[k] = v[-1]
            elif self.training:
                final_results[k] = torch.stack(v).mean()
        return final_results

class ResShortCut_AttenSpconv_BiTempSpar_Dec(ResShortCut_AttenSpconv_Dec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # self.os8_temp_module = WindowSTM(128, os=8, mask_channel=kwargs["embed_dim"])
        self.os16_temp_module = ConvGRU(256)
        self.os8_temp_module = ConvGRU(128)

        # Module to compute the difference between two inputs
        self.diff_module = nn.Sequential(
            SpectralNorm(conv1x1(256, 64)),
            self._norm_layer(64),
            nn.ReLU(inplace=True),
            SpectralNorm(conv3x3(64, 32)),
            self._norm_layer(32),
            nn.ReLU(inplace=True),
            conv3x3(32, 1)
        )

    def bidirectional_fusion(self, feat, preds):
        '''
        Fuse forward and backward features
        '''
        n_f = feat.shape[1]
        forward_diffs = []
        backward_diffs = []
        forward_preds = [preds[:, 0]]
        backward_preds = [preds[:, n_f-1]]
        fuse_preds = []

        # forward
        for i in range(1, n_f):
            diff = self.diff_module(torch.cat([feat[:, i-1], feat[:, i]], dim=1))
            diff = F.interpolate(diff, scale_factor=8.0, mode='bilinear', align_corners=False)
            forward_diffs.append(diff)
            pred = forward_preds[-1] * (1 - diff.sigmoid()) + preds[:, i] * diff.sigmoid()
            forward_preds.append(pred)
        
        forward_diffs = [torch.zeros_like(forward_diffs[0])] + forward_diffs
        forward_diffs = torch.stack(forward_diffs, dim=1)

        # backward
        for i in range(n_f-1, 0, -1):
            diff = self.diff_module(torch.cat([feat[:, i], feat[:, i-1]], dim=1))
            diff = F.interpolate(diff, scale_factor=8.0, mode='bilinear', align_corners=False)
            backward_diffs.append(diff)
            pred = backward_preds[-1] * (1 - diff.sigmoid()) + preds[:, i-1] * diff.sigmoid()
            backward_preds.append(pred)
        
        backward_preds = backward_preds[::-1]
        backward_diffs = backward_diffs[::-1]
        backward_diffs = backward_diffs + [torch.zeros_like(backward_diffs[-1])]
        backward_diffs = torch.stack(backward_diffs, dim=1)

        # Fuse forward and backward
        for i in range(n_f):
            if i == 0:
                fuse_preds.append(forward_preds[i])
            elif i == n_f - 1:
                fuse_preds.append(backward_preds[i])
            else:
                fuse_preds.append((forward_preds[i] + backward_preds[i]) / 2)
        fuse_preds = torch.stack(fuse_preds, dim=1)
        return forward_diffs, backward_diffs, fuse_preds
    
    def forward(self, x, mid_fea, b, n_f, n_i, masks, iter, gt_alphas, mem_feat=None, mem_query=None, mem_details=None, spar_gt=None, **kwargs):
        
        '''
        masks: [b * n_f * n_i, 1, H, W]
        '''
        # Reshape masks
        masks = masks.reshape(b, n_f, n_i, masks.shape[2], masks.shape[3])
        valid_masks = masks.flatten(0,1).sum((2, 3), keepdim=True) > 0
        gt_masks = None
        if self.training:
            gt_masks = (gt_alphas > 0).reshape(b, n_f, n_i, gt_alphas.shape[2], gt_alphas.shape[3])

        mem_os16, mem_os8 = None, None
        if mem_feat is not None:
            mem_os16, mem_os8 = mem_feat

        # OS32 -> OS 8
        ret = {}
        fea4, fea5 = mid_fea['shortcut']
        
        image = mid_fea['image']
        x = self.layer1(x) + fea5

        # Temporal aggregation
        x = x.view(b, n_f, *x.shape[1:])
        x, mem_os16 = self.os16_temp_module(x, mem_os16)
        x = x.view(b * n_f, *x.shape[2:])
        # import pdb; pdb.set_trace()

        x = self.layer2(x) + fea4

        # feat os8 to compute the differences
        # feat_os8 = x.view(b, n_f, *x.shape[1:])

        # Perform temporal aggregation here: 
        # x = x.view(b, n_f, *x.shape[1:])
        # feat0 = x[:, 0]
        # mem_feats = []
        # if mem_feat is not None:
        #     feat0 = self.os8_temp_module(x[:, 0], mem_feat)
        #     mem_feats.append(mem_feat)
        # mem_feats.append(feat0[:, None])
        # feat1 = self.os8_temp_module(x[:, 1], torch.cat(mem_feats, dim=1))
        # mem_feats.append(feat1[:, None])
        # feat2 = self.os8_temp_module(x[:, 2], torch.cat(mem_feats, dim=1))

        # x = torch.cat([feat0[:, None], feat1[:, None], feat2[:, None]], dim=1)
        # x = x.view(b * n_f, *x.shape[2:])

        # For ConvGRU
        x = x.view(b, n_f, *x.shape[1:])
        x, mem_os8 = self.os8_temp_module(x, mem_os8)
        x = x.view(b * n_f, *x.shape[2:])
        # import pdb; pdb.set_trace()
        
        mem_feat = (mem_os16, mem_os8)

        # Predict OS8
        # use mask attention during warmup of training
        x_os8, x, queries, loss_max_atten, loss_min_atten = self.refine_OS8(x, masks, 
                                                                            prev_tokens=mem_query if self.use_query_temp else None, 
                                                                            use_mask_atten=False, gt_mask=gt_masks)

        # Predict temporal sparsity here, forward and backward
        feat_os8 = x.view(b, n_f, *x.shape[1:])


        # diff_forward01 = self.diff_module(torch.cat([feat_os8[:, 0], feat_os8[:, 1]], dim=1))
        # diff_forward12 = self.diff_module(torch.cat([feat_os8[:, 1], feat_os8[:, 2]], dim=1))
        # diff_backward21 = self.diff_module(torch.cat([feat_os8[:, 2], feat_os8[:, 1]], dim=1))

        # Upscale diff
        # diff_forward = F.interpolate(diff_forward, scale_factor=8.0, mode='bilinear', align_corners=False)
        # diff_backward = F.interpolate(diff_backward, scale_factor=8.0, mode='bilinear', align_corners=False)
            

        # Upsample - normalize OS8 pred
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0
        if self.training:
            x_os8 = x_os8 * valid_masks
        else:
            x_os8 = x_os8[:, :n_i]

        # Smooth features
        x = self.layer3(x)

        # Warm-up - Using gt_alphas instead of x_os8 for later steps
        guided_mask_os8 = x_os8
        if self.training and (iter < self.warmup_detail_iter or x_os8.sum() == 0 or (iter < self.warmup_detail_iter * 3 and random.random() < 0.5)):
            guided_mask_os8 = gt_alphas.clone()
        
        # Compute unknown regions
        unknown_os8 = compute_unknown(guided_mask_os8, k_size=58, is_train=False)

        # Dummy code to prevent all zeros
        if unknown_os8.max() == 0 and self.training:
            unknown_os8[:, :, 200: 250, 200: 250] = 1.0

        # Predict details
        if unknown_os8.sum() > 0 or self.training:
            x_os4, x_os1, mem_details = self.predict_details(x, image, unknown_os8, guided_mask_os8, mem_details)
            x_os4 = x_os4.reshape(b * n_f, guided_mask_os8.shape[1], *x_os4.shape[-2:])
            x_os1 = x_os1.reshape(b * n_f, guided_mask_os8.shape[1], *x_os1.shape[-2:])

            x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
            x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
            x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        else:
            x_os4 = torch.zeros((b * n_f, x_os8.shape[1], image.shape[2], image.shape[3]), device=x_os8.device)
            x_os1 = torch.zeros_like(x_os4)
        
        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8
        ret['mem_feat'] = mem_feat

        # Fusion
        alpha_pred, _, _ = self.fushion(ret, unknown_os8)

        # Fuse temporal sparsity
        temp_alpha = alpha_pred.view(b, n_f, *alpha_pred.shape[1:])
        # temp_forward = temp_alpha[:, 0] * (1 - diff_forward.sigmoid()) + temp_alpha[:, 1] * diff_forward.sigmoid()
        # temp_backward = temp_alpha[:, 2] * (1 - diff_backward.sigmoid()) + temp_alpha[:, 1] * diff_backward.sigmoid()
        # temp_fused_alpha = (temp_forward + temp_backward) / 2.0

        # if not self.training:
        #     temp_alpha[:, 1] = temp_fused_alpha
        diff_forward, diff_backward, temp_fused_alpha = self.bidirectional_fusion(feat_os8, temp_alpha)

        ret['refined_masks'] = alpha_pred
        ret['detail_mask'] = unknown_os8

        ret['temp_alpha'] = temp_fused_alpha
        ret['diff_forward'] = diff_forward.sigmoid()
        ret['diff_backward'] = diff_backward.sigmoid()

        # Adding some losses to the results
        if self.training:
            ret['loss_max_atten'] = loss_max_atten
            ret['loss_min_atten'] = loss_min_atten

            # Compute loss for temporal sparsity
            ret.update(self.loss_temporal_sparsity(diff_forward, diff_backward, spar_gt, gt_alphas))

        return ret
    
    def loss_temporal_sparsity(self, diff_forward, diff_backward, spar_gt, alphas_gt):
        '''
        Compute loss for temporal sparsity
        '''
        loss = {}

        # BCE loss
        spar_gt = spar_gt.view(diff_forward.shape[0], -1, *spar_gt.shape[1:])
        bce_forward_loss = F.binary_cross_entropy_with_logits(diff_forward[:, 1:, 0], spar_gt[:, 1:, 0], reduction='mean')
        bce_backward_loss = F.binary_cross_entropy_with_logits(diff_backward[:, :-1, 0], spar_gt[:, 1:, 0], reduction='mean')
        # bce_loss = F.binary_cross_entropy_with_logits(torch.cat([diff_forward, diff_backward], dim=1), spar_gt[:, 1:, 0], reduction='mean')
        loss['loss_temp_bce'] = bce_forward_loss + bce_backward_loss

        # Fusion loss
        
        alphas_gt = alphas_gt.view(diff_forward.shape[0], -1, *alphas_gt.shape[1:])
        temp_forward = alphas_gt[:, 0] * (1 - diff_forward[:, 1].sigmoid()) + alphas_gt[:, 1] * diff_forward[:, 1].sigmoid()
        temp_backward = alphas_gt[:, 2] * (1 - diff_backward[:, 1].sigmoid()) + alphas_gt[:, 1] * diff_backward[:, 1].sigmoid()
        fusion_loss = F.l1_loss(temp_forward, temp_backward, reduction='none') + \
            F.l1_loss(temp_forward, alphas_gt[:, 1], reduction='none') + \
                F.l1_loss(temp_backward, alphas_gt[:, 1], reduction='none')
        
        valid_masks = alphas_gt.sum((3, 4), keepdim=True)[:, 1] > 0
        valid_masks = valid_masks.repeat(1, 1, *fusion_loss.shape[2:])
        fusion_loss = fusion_loss[valid_masks].mean()
        loss['loss_temp_fusion'] = fusion_loss

        loss['loss_temp'] = loss['loss_temp_bce'] * 0.25 + fusion_loss

        return loss

def res_shortcut_attention_spconv_decoder_22(**kwargs):
    return ResShortCut_AttenSpconv_Dec(BasicBlock, [2, 3, 3, 2], **kwargs)

def res_shortcut_attention_spconv_temp_decoder_22(**kwargs):
    return ResShortCut_AttenSpconv_Temp_Dec(block=BasicBlock, layers=[2, 3, 3, 2], **kwargs)

def res_shortcut_attention_spconv_querytemp_decoder_22(**kwargs):
    return ResShortCut_AttenSpconv_QueryTemp_Dec(block=BasicBlock, layers=[2, 3, 3, 2], **kwargs)

def res_shortcut_attention_spconv_inconsisttemp_decoder_22(**kwargs):
    return ResShortCut_AttenSpconv_InconsistTemp_Dec(block=BasicBlock, layers=[2, 3, 3, 2], **kwargs)

def res_shortcut_attention_spconv_bitempspar_decoder_22(**kwargs):
    return ResShortCut_AttenSpconv_BiTempSpar_Dec(block=BasicBlock, layers=[2, 3, 3, 2], **kwargs)