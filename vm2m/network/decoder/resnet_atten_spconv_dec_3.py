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
from vm2m.utils.utils import compute_unknown, resizeAnyShape, gaussian_smoothing
from .resnet_dec import BasicBlock
from vm2m.network.loss import loss_dtSSD

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
                  use_id_pe=True, use_query_temp=False, use_detail_temp=False, detail_mask_dropout=0.2, warmup_detail_iter=3000, \
                    use_temp=False, freeze_detail_branch=False, **kwargs):
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
            use_temp=use_temp
        )
        # relu_layer = nn.ReLU(inplace=True)

        # Instance feature guidance at OS8
        # self.fg_fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.Sigmoid()) # OS8, FC -> ReLU
        # self.bg_fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True)) # OS8, FC -> ReLU
        # self.guidance_fc = nn.Sequential(nn.Linear(256, 128), nn.Sigmoid()) # OS8
        
        # Modules to save the sampling path of sparse Conv
        self.dummy_downscale = spconv.SparseSequential(
            spconv.SubMConv2d(3, 32, kernel_size=3, padding=1, bias=False, indice_key="subminp"),
            spconv.SparseConv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False, indice_key="subm1.2"),
            spconv.SparseConv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False, indice_key="subm2.4"),
            spconv.SparseConv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False, indice_key="subm4.8"),
        )

        # Sparse Conv from OS8 to OS4
        self.layer3 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(64, 64, kernel_size=3, bias=False, indice_key="subm4.8"),
            nn.BatchNorm1d(64),
            self.leaky_relu,
            spconv.SubMConv2d(64, 64, kernel_size=3, padding=1, bias=False, indice_key="subm4.4"),
        )

        self.layer3_smooth = spconv.SparseSequential(
            spconv.SubMConv2d(128, 64, kernel_size=1, padding=0, bias=True, indice_key="subm4.smooth"),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        # Sparse Conv from OS4 to OS2
        self.layer4 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(64, 32, kernel_size=3, bias=False, indice_key="subm2.4"),
            nn.BatchNorm1d(32),
            self.leaky_relu,
            spconv.SubMConv2d(32, 32, kernel_size=1, padding=1, bias=False, indice_key="subm2.2"),
        )

        self.layer4_smooth = spconv.SparseSequential(
            spconv.SubMConv2d(64, 32, kernel_size=1, padding=0, bias=True, indice_key="subm2.smooth"),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
        )

        # Sparse Conv from OS2 to OS1
        self.layer5 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(32, 32, kernel_size=3, bias=False, indice_key="subm1.2"),
            nn.BatchNorm1d(32),
            self.leaky_relu,
            spconv.SubMConv2d(32, 32, kernel_size=3, padding=1, bias=False, indice_key="subm1.1"),
        )

        self.layer5_smooth = spconv.SparseSequential(
            spconv.SubMConv2d(64, 32, kernel_size=1, padding=0, bias=True, indice_key="subm1.smooth"),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
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

        self.freeze_detail_branch = freeze_detail_branch
        self.train()

    def freeze_coarse_layers(self):
        coarse_modules = [self.layer1, self.layer2, self.refine_OS8]
        for module in coarse_modules:
            for p in module.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super(ResShortCut_AttenSpconv_Dec, self).train(mode)
        if mode and self.freeze_detail_branch:
            detail_modules = [self.layer3, self.layer4, self.layer5, self.low_os1_module, self.low_os2_module, self.low_os4_module, self.refine_OS4, self.refine_OS8]
            for module in detail_modules:
                for p in module.parameters():
                    p.requires_grad = False

        for p in self.dummy_downscale.parameters():
            p.requires_grad = False

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
    
    def combine_dense_sparse_feat(self, sparse_feat, dense_feat, b, n_i, h, w):
        coords = sparse_feat.indices.clone()
        coords[:, 0] = torch.div(coords[:, 0], n_i, rounding_mode='floor')
        coords = coords.long()
        x = dense_feat.permute(0, 2, 3, 1).contiguous()
        x = x[coords[:, 0], coords[:, 1], coords[:, 2]]

        x = spconv.SparseConvTensor(x, sparse_feat.indices, (h, w), b * n_i, indice_dict=sparse_feat.indice_dict)
        x = x.replace_feature(torch.cat([x.features, sparse_feat.features], dim=1))
        # x = Fsp.sparse_add(x, sparse_feat)
        return x
    
    def predict_details(self, os8_feat, image, roi_masks, masks, mem_details=None, inst_guidance_os8=None, dense_features=None):
        '''
        x: [b, 64, h/4, w/4], os4 semantic features
        image: [b, 3, H, W], original image
        masks: [b, n_i, H, W], dilated guided masks from OS8 prediction
        mem_details: SparseConvTensor, memory details
        '''
        # Stack masks and images
        b, n_i, H, W = masks.shape
        roi_masks = roi_masks.reshape(b * n_i, 1, H, W)

        coords = torch.nonzero(roi_masks.squeeze(1) > 0, as_tuple=True)

        image_batch_indices = torch.div(coords[0], n_i, rounding_mode='floor')
        image_vals = image[image_batch_indices, :, coords[1], coords[2]]
        
        inp = image_vals
        
        coords = torch.stack(coords, dim=1)
        inp = spconv.SparseConvTensor(inp, coords.int(), (H, W), b * n_i)

        # Build sampling paths
        with torch.no_grad():
            dummy_os8 = self.dummy_downscale(inp)

        # Get sparse features of OS8 feat
        coords = dummy_os8.indices.clone()
        coords[:, 0] = torch.div(coords[:, 0], n_i, rounding_mode='floor') # batch ids
        coords = coords.long()
        x = os8_feat[coords[:, 0], :, coords[:, 1], coords[:, 2]]
        x = spconv.SparseConvTensor(x, dummy_os8.indices, (H // 8, W // 8), b * n_i, indice_dict=dummy_os8.indice_dict)

        # Change image features to instance-specific features by instance guidance
        instance_ids = x.indices[:, 0] % n_i
        # import pdb; pdb.set_trace()
        guidance = inst_guidance_os8[coords[:, 0], instance_ids.long()]
        x = x.replace_feature(x.features * guidance)

        # Visualize feature before-after guidance
        # for i in valid_ids:
        #     if i == 0: continue
        #     print(i)
        #     b_i = i // n_i
        #     i_i = i % n_i
        #     feat_m_before = os8_feat[b_i]
        #     cv2.imwrite("roi_mask.png", roi_masks[i, 0].cpu().numpy() * 255)
        #     for j in range(128):
        #         print(inst_guidance_os8[b_i, i_i, j])
        #         feat_m = feat_m_before[j]
        #         feat_m = (feat_m - feat_m.min()) / (feat_m.max() - feat_m.min() + 1e-8)
        #         cv2.imwrite("feat_map.png", feat_m.detach().float().cpu().numpy() * 255)
        #         import pdb; pdb.set_trace()
        #     feat_m_after = os8_feat[b_i] * inst_guidance_os8[b_i, i_i][:, None, None]
        #     feat_m_before = feat_m_before.mean(0)
        #     feat_m_after = feat_m_after.mean(0)
        #     feat_m_before = (feat_m_before - feat_m_before.min()) / (feat_m_before.max() - feat_m_before.min() + 1e-8)
        #     feat_m_after = (feat_m_after - feat_m_after.min()) / (feat_m_after.max() - feat_m_after.min() + 1e-8)
        #     cv2.imwrite("feat_map_before.png", feat_m_before.detach().float().cpu().numpy() * 255)
        #     cv2.imwrite("feat_map_after.png", feat_m_after.detach().float().cpu().numpy() * 255)
        #     cv2.imwrite("roi_mask.png", roi_masks[i, 0].cpu().numpy() * 255)
        #     import pdb; pdb.set_trace()
        # for i in valid_ids:
        #     print(i)
        #     feat_m = dense_feat[i].mean(0)
        #     feat_m = (feat_m - feat_m.min()) / (feat_m.max() - feat_m.min() + 1e-8)
        #     cv2.imwrite("feat_map.png", feat_m.detach().float().cpu().numpy() * 255)
        #     cv2.imwrite("roi_mask.png", roi_masks[i, 0].cpu().numpy() * 255)
        #     import pdb; pdb.set_trace()
        
        # Aggregate with detail features
        fea1, fea2, fea3 = dense_features
        # Aggregate with OS4 features
        # import pdb; pdb.set_trace()
        x = self.layer3(x)
        # import pdb; pdb.set_trace()
        x = self.combine_dense_sparse_feat(x, fea3, b, n_i, H // 4, W // 4)
        x = self.layer3_smooth(x)

        # TODO: Check instance specific features here
        # dense_feat = x.dense()
        # valid_ids = torch.nonzero(roi_masks.sum((1,2,3))).flatten()
        # for i in valid_ids:
        #     print(i)
        #     feat_m = dense_feat[i].mean(0)
        #     feat_m = (feat_m - feat_m.min()) / (feat_m.max() - feat_m.min() + 1e-8)
        #     cv2.imwrite("feat_map.png", feat_m.detach().float().cpu().numpy() * 255)
        #     import pdb; pdb.set_trace()

        # Predict OS 4
        x_os4 = self.refine_OS4(x)
        x_os4_out = x_os4.dense()
        x_os4_out = x_os4_out - 99
        coords = x_os4.indices.long()
        x_os4_out[coords[:, 0], :, coords[:, 1], coords[:, 2]] += 99

        # Aggregate with OS2 features
        x = self.layer4(x)
        x = self.combine_dense_sparse_feat(x, fea2, b, n_i, H // 2, W // 2)
        x = self.layer4_smooth(x)

        # Aggregate with OS1 features
        x = self.layer5(x)

        
        # TODO: Check instance specific features here
        x = self.combine_dense_sparse_feat(x, fea1, b, n_i, H, W)
        x = self.layer5_smooth(x)
        

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
        weight_os4 = compute_unknown(alpha_pred, k_size=27, is_train=self.training) * detail_mask
        weight_os4 = weight_os4.type(alpha_pred.dtype)
        alpha_pred_os4 = alpha_pred_os4.type(alpha_pred.dtype)
        weight_os4 = weight_os4 #* (alpha_pred_os4 > 1.0/255).float()
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4> 0]
        # cv2.imwrite("test_fuse.png", alpha_pred[0,0].detach().cpu().numpy() * 255)
        # cv2.imwrite("test_fuse_wos4.png", weight_os4[0,0].detach().cpu().numpy() * 255)
        # cv2.imwrite("test_fuse_os8.png", alpha_pred_os8[0,0].detach().cpu().numpy() * 255)
        # cv2.imwrite("test_fuse_os4.png", alpha_pred_os4[0,0].detach().cpu().numpy() * 255)
        # import pdb; pdb.set_trace()

        # weight_os1 = get_unknown_tensor_from_pred(alpha_pred, rand_width=15, train_mode=self.training)
        weight_os1 = compute_unknown(alpha_pred, k_size=15, is_train=self.training) * detail_mask
        weight_os1 = weight_os1.type(alpha_pred.dtype)
        alpha_pred_os1 = alpha_pred_os1.type(alpha_pred.dtype)
        weight_os1 = weight_os1 #* (alpha_pred_os1 > 1.0/255).float()
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1 > 0]

        # Blending between 1.0 and < 1.0 region
        # fg_mask = (alpha_pred >= 254.0/255.0).float()
        # smooth_alpha_pred = gaussian_smoothing(alpha_pred, sigma=1)
        # alpha_pred = fg_mask * smooth_alpha_pred + (1 - fg_mask) * alpha_pred

        return alpha_pred, weight_os4, weight_os1

    def compute_instance_guidance(self, feat, guidance_map):
        # import pdb; pdb.set_trace()
        # bg_guidance = (1.0 - guidance_map.sum(1)).clamp(0, 1)
        # bg_guidance = bg_guidance[:, None].expand_as(guidance_map)
        fg_guidance = F.adaptive_avg_pool2d(guidance_map[:, :, None] * feat[:, None], 1).squeeze(-1).squeeze(-1)
        # fg_guidance = (guidance_map[:, :, None] * feat[:, None]).sum((-1, -2)) / (guidance_map[:, :, None].sum((-1, -2)) + 1e-8)
        # bg_guidance = F.adaptive_avg_pool2d((1 - guidance_map[:, :, None]) * feat[:, None], 1).squeeze(-1).squeeze(-1)
        # bg_guidance = (bg_guidance[:, :, None] * feat[:, None]).sum((-1, -2)) / (bg_guidance[:, :, None].sum((-1, -2)) + 1e-8)
        fg_guidance = self.fg_fc(fg_guidance)
        return fg_guidance
        # bg_guidance = self.bg_fc(bg_guidance)
        # instance_guidance = self.guidance_fc(torch.cat([fg_guidance, bg_guidance], dim=2))
        # return instance_guidance
    
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

            if gt_masks.shape[-1] != masks.shape[-1]:
                gt_masks = resizeAnyShape(gt_masks, scale_factor=masks.shape[-1] * 1.0/ gt_masks.shape[-1], use_max_pool=True)

        # OS32 -> OS 8
        ret = {}
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        
        image = mid_fea['image']
        x = self.layer1(x) + fea5
        
        # Update with memory
        # x = self.aggregate_mem(x, mem_feat)
        # mem_feat = x

        x = self.layer2(x) + fea4

        h, w = image.shape[-2:]

        # Predict OS8
        # import pdb; pdb.set_trace()
        # use mask attention during warmup of training
        use_mask_atten = iter < self.warmup_mask_atten_iter and self.training
        # import pdb; pdb.set_trace()
        x_os8, x, _, queries, loss_max_atten, loss_min_atten = self.refine_OS8(x, masks, 
                                                                            prev_tokens=mem_query if self.use_query_temp else None, 
                                                                            use_mask_atten=use_mask_atten, gt_mask=gt_masks)
        mem_feat = x
        x_os8 = F.interpolate(x_os8, size=(h, w), mode='bilinear', align_corners=False)
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0
        if self.training:
            x_os8 = x_os8 * valid_masks
        else:
            x_os8 = x_os8[:, :n_i]


        # Stop when freeze detail branch
        if self.freeze_detail_branch:
            ret['alpha_os8'] = x_os8
            ret['refined_masks'] = x_os8
            ret['loss_max_atten'] = loss_max_atten
            ret['loss_min_atten'] = loss_min_atten
            return ret
        
        # Warm-up - Using gt_alphas instead of x_os8 for later steps
        guided_mask_os8 = x_os8.clone()
        is_use_alphas_gt = False
        if self.training and (iter < self.warmup_detail_iter or x_os8.sum() == 0 or (iter < self.warmup_detail_iter * 3 and random.random() < 0.5)):
            guided_mask_os8 = gt_alphas.clone()
            is_use_alphas_gt = True
            # guided_mask_os8 = F.interpolate(guided_mask_os8, scale_factor=0.0625, mode='bilinear', align_corners=False)
            # guided_mask_os8 = F.interpolate(guided_mask_os8, size=x_os8.shape[-2:], mode='bilinear', align_corners=False)

        # Compute instance guidance
        # guidance_map_os8 = F.interpolate(guided_mask_os8, scale_factor=0.125, mode='bilinear', align_corners=False)
        # inst_guidance_os8 = self.compute_instance_guidance(x, guidance_map_os8)
        # import pdb; pdb.set_trace()
        # Apply gaussian filter on guided_mask_os8
        # guided_mask_os8 = gaussian_smoothing(guided_mask_os8, sigma=3)
        unknown_os8 = compute_unknown(guided_mask_os8, k_size=30)

        if unknown_os8.max() == 0 and self.training:
            unknown_os8[:, :, 200: 250, 200: 250] = 1.0

        if unknown_os8.sum() > 0 or self.training:

            x_os4, x_os1, mem_details = self.predict_details(x, image, unknown_os8, guided_mask_os8, mem_details, queries, [fea1, fea2, fea3])

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

        # diff_os4 = torch.abs(x_os4 - x_os8 * unknown_os8)
        # diff_os1 = torch.abs(x_os1 - x_os8 * unknown_os8)
        # cv2.imwrite("test_diff_os4.png", diff_os4[0,1].detach().cpu().numpy() * 255)
        # cv2.imwrite("test_diff_os1.png", diff_os1[0,1].detach().cpu().numpy() * 255)
        # import pdb; pdb.set_trace()
        # if not self.training:
        #     import pdb; pdb.set_trace()
        # Update mem_feat
        alpha_pred, weight_os4, weight_os1 = self.fushion(ret, unknown_os8)
        # cv2.imwrite("test_fuse_new.png", alpha_pred[0, 1].cpu().numpy() * 255)
        # import pdb; pdb.set_trace()
        # mem_feat = self.update_mem(mem_feat, alpha_pred)
        # mem_feat = (mem_feat, prev_pred)

        # Update mem_details
        mem_details = self.update_detail_mem(mem_details, alpha_pred)

        ret['refined_masks'] = alpha_pred
        # ret['refined_masks'] = x_os8
        
        # If we use GT alphas, select amount randomly in the unknown mask region
        if is_use_alphas_gt:
            weight_os4 = compute_unknown(gt_alphas, k_size=30, is_train=self.training) * unknown_os8
            weight_os1 = compute_unknown(gt_alphas, k_size=15, is_train=self.training) * unknown_os8
        
        ret['weight_os4'] = weight_os4
        ret['weight_os1'] = weight_os1 
        ret['detail_mask'] = unknown_os8
        # ret['diff_pred'] = diff_pred
        if self.training and iter >= self.warmup_mask_atten_iter:
            ret['loss_max_atten'] = loss_max_atten
            ret['loss_min_atten'] = loss_min_atten
        ret['mem_queries'] = queries
        ret['mem_feat'] = mem_feat
        ret['mem_details'] = mem_details
        return ret

class ResShortCut_AttenSpconv_BiTempSpar_Dec(ResShortCut_AttenSpconv_Dec):
    def __init__(self, temp_method='bi', **kwargs):
        super().__init__(use_temp=True, **kwargs)
        
        self.temp_method = temp_method.split("_")[0]
        self.use_fusion = 'fusion' in temp_method
        self.use_temp = temp_method != 'none'

        self.os8_temp_module = ConvGRU(128, dilation=1, padding=1)

        # Module to compute the difference between two inputs
        self.diff_module = nn.Sequential(
            SpectralNorm(conv1x1(128, 64)),
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

        # OS32 -> OS 8
        ret = {}
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        
        image = mid_fea['image']
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        
        x_os8, x, hidden_state, queries, loss_max_atten, loss_min_atten = self.refine_OS8(x, masks, 
                                                                            prev_tokens=mem_query if self.use_query_temp else None, 
                                                                            use_mask_atten=False, gt_mask=gt_masks, 
                                                                            aggregate_mem_fn=self.os8_temp_module.forward, prev_h_state=mem_feat, temp_method=self.temp_method)
        
        mem_feat = hidden_state

        # Predict temporal sparsity here, forward and backward
        feat_os8 = x.view(b, n_f, *x.shape[1:])
            
        # Upsample - normalize OS8 pred
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

        if self.training:
            x_os8 = x_os8 * valid_masks
        else:
            x_os8 = x_os8[:, :n_i]


        # Warm-up - Using gt_alphas instead of x_os8 for later steps
        guided_mask_os8 = x_os8
        is_use_alphas_gt = False
        if self.training and (iter < self.warmup_detail_iter or x_os8.sum() == 0 or (iter < self.warmup_detail_iter * 3 and random.random() < 0.5)):
            guided_mask_os8 = gt_alphas.clone()
            is_use_alphas_gt = True
        
        # Compute unknown regions
        unknown_os8 = compute_unknown(guided_mask_os8, k_size=30)

        # Dummy code to prevent all zeros
        if unknown_os8.max() == 0 and self.training:
            unknown_os8[:, :, 200: 250, 200: 250] = 1.0
        
        # Expand queries to N_F
        queries = queries[:, None].expand(-1, n_f, -1, -1).reshape(b * n_f, *queries.shape[1:])

        # Predict details
        if unknown_os8.sum() > 0 or self.training:
            x_os4, x_os1, mem_details = self.predict_details(x, image, unknown_os8, guided_mask_os8, mem_details, queries, [fea1, fea2, fea3])
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

        # Fusion
        alpha_pred, weight_os4, weight_os1 = self.fushion(ret, unknown_os8)

        # Fuse temporal sparsity
        temp_alpha = alpha_pred.view(b, n_f, *alpha_pred.shape[1:])

        if self.use_temp:
            ret['mem_feat'] = mem_feat
        
        
        diff_forward, diff_backward, temp_fused_alpha = self.bidirectional_fusion(feat_os8, temp_alpha)

        ret['refined_masks'] = alpha_pred
        ret['detail_mask'] = unknown_os8

        if is_use_alphas_gt:
            weight_os4 = compute_unknown(gt_alphas, k_size=30, is_train=self.training) * unknown_os8
            weight_os1 = compute_unknown(gt_alphas, k_size=15, is_train=self.training) * unknown_os8

        ret['weight_os4'] = weight_os4
        ret['weight_os1'] = weight_os1 

        # import pdb; pdb.set_trace()
        if (not self.training and self.use_fusion) or self.training:
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
        # import pdb; pdb.set_trace()
        dtSSD_forward = loss_dtSSD(diff_forward[:, 1:].sigmoid(), spar_gt[:, 1:, 0:1], torch.ones_like(spar_gt[:, 1:, 0:1]))
        dtSSD_backward = loss_dtSSD(diff_backward[:, :-1].sigmoid(), spar_gt[:, 1:, 0:1], torch.ones_like(spar_gt[:, 1:, 0:1]))
        loss['loss_temp_dtssd'] = dtSSD_forward + dtSSD_backward

        loss['loss_temp'] = (loss['loss_temp_bce'] + dtSSD_forward + dtSSD_backward) * 0.25

        return loss

def res_shortcut_attention_spconv_decoder_22(**kwargs):
    return ResShortCut_AttenSpconv_Dec(BasicBlock, [2, 3, 3, 2], **kwargs)

def res_shortcut_attention_spconv_bitempspar_decoder_22(**kwargs):
    return ResShortCut_AttenSpconv_BiTempSpar_Dec(block=BasicBlock, layers=[2, 3, 3, 2], **kwargs)