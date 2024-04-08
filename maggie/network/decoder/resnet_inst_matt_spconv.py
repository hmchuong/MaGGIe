import random

import torch
from torch import nn
from torch.nn import functional as F

import spconv.pytorch as spconv

from .resnet import BasicBlock
from ..module import SpectralNorm, conv1x1, InstanceMatteDecoder
from ..module.mask_attention import FFNLayer
from ...utils.utils import compute_unknown, resizeAnyShape

class ResShortCut_InstMattSpconv_Dec(nn.Module):
    def __init__(self, block, layers, 
                 norm_layer=None, large_kernel=False, late_downsample=False, 
                 atten_stride=1, atten_dim=128, atten_block=2, atten_head=1, final_channel=32,  max_inst=10, use_id_pe=True, 
                 warmup_mask_atten_iter=4000, warmup_detail_iter=3000, use_query_temp=False, use_detail_temp=False, detail_mask_dropout=0.2, **kwargs):
        super(ResShortCut_InstMattSpconv_Dec, self).__init__()
        
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

        # Instance-Spec features
        self.inst_spec_layer = FFNLayer(final_channel, final_channel, 0.1)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.tanh = nn.Tanh()
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

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
        
        # Modules to save the sampling path of sparse Conv
        self.dummy_downscale = spconv.SparseSequential(
            spconv.SubMConv2d(3, 32, kernel_size=3, padding=1, bias=False, indice_key="subminp"),
            spconv.SparseConv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False, indice_key="subm1.2"),
            spconv.SparseConv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False, indice_key="subm2.4"),
            spconv.SparseConv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False, indice_key="subm4.8"),
        )

        # Sparse Conv from OS8 to OS4
        self.layer3 = spconv.SparseSequential(
            spconv.SparseInverseConv2d(final_channel, 64, kernel_size=3, bias=False, indice_key="subm4.8"),
            nn.BatchNorm1d(64),
            self.leaky_relu,
            spconv.SubMConv2d(64, 64, kernel_size=3, padding=1, bias=False, indice_key="subm4.4"),
        )

        self.guidance_layer = spconv.SparseSequential(
            spconv.SubMConv2d(128, 64, kernel_size=1, padding=0, bias=False, indice_key="subm_inst.0"),
            nn.BatchNorm1d(64),
            self.leaky_relu,
            spconv.SubMConv2d(64, 64, kernel_size=3, padding=1, bias=True, indice_key="subm_inst.1"),
            nn.Sigmoid()
        )

        self.layer3_smooth = spconv.SparseSequential(
            spconv.SubMConv2d(64, 64, kernel_size=1, padding=0, bias=True, indice_key="subm4.smooth"),
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
    
    
    def combine_dense_sparse_feat(self, sparse_feat, dense_feat, b, n_i, h, w):
        coords = sparse_feat.indices.clone()
        coords[:, 0] = torch.div(coords[:, 0], n_i, rounding_mode='floor')
        coords = coords.long()
        x = dense_feat.permute(0, 2, 3, 1).contiguous()
        x = x[coords[:, 0], coords[:, 1], coords[:, 2]]

        x = spconv.SparseConvTensor(x, sparse_feat.indices, (h, w), b * n_i, indice_dict=sparse_feat.indice_dict)
        x = x.replace_feature(torch.cat([x.features, sparse_feat.features], dim=1))
        return x

    def instance_spec_guidance(self, sparse_feat, dense_feat, b, n_i, h, w):
        '''
        sparse_feat: SparseConvTensor, instance-specific features
        dense_feat: dense detail features
        '''
        coords = sparse_feat.indices.clone()
        coords[:, 0] = torch.div(coords[:, 0], n_i, rounding_mode='floor')
        coords = coords.long()
        x = dense_feat.permute(0, 2, 3, 1).contiguous()
        x = x[coords[:, 0], coords[:, 1], coords[:, 2]]

        # Detail sparse features
        x = spconv.SparseConvTensor(x, sparse_feat.indices, (h, w), b * n_i, indice_dict=sparse_feat.indice_dict)
        
        # Predict coarse guidance
        detail_features = x.features
        x = x.replace_feature(torch.cat([x.features, sparse_feat.features], dim=1))
        guidance = self.guidance_layer(x)

        # Guide detail features with guidance
        x = x.replace_feature(detail_features * guidance.features)

        return x
    
    def predict_details(self, os8_feat, image, roi_masks, masks, inst_guidance_os8, dense_features):
        '''
        x: [b, 64, h/4, w/4], os4 semantic features
        image: [b, 3, H, W], original image
        masks: [b, n_i, H, W], dilated guided masks from OS8 prediction
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
        guidance = inst_guidance_os8[coords[:, 0], instance_ids.long()]
        
        # Get instance-spec features at OS8
        x = x.replace_feature(self.inst_spec_layer(x.features * guidance))
        
        # Aggregate with detail features
        fea1, fea2, fea3 = dense_features
        
        # Aggregate with OS4 features
        x = self.layer3(x)

        # Instance-spec guidance detail feat
        # x: os8 instance spec feat, fea3: detail feat
        x = self.instance_spec_guidance(x, fea3, b, n_i, H // 4, W // 4)
        
        x = self.layer3_smooth(x)

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
        x = self.combine_dense_sparse_feat(x, fea1, b, n_i, H, W)
        x = self.layer5_smooth(x)
        
        # Predict OS 1
        x_os1 = self.refine_OS1(x)
        x_os1_out = x_os1.dense()
        x_os1_out = x_os1_out - 99
        coords = x_os1.indices.long()
        x_os1_out[coords[:, 0], :, coords[:, 1], coords[:, 2]] += 99

        return x_os4_out, x_os1_out

    def fuse(self, pred, detail_mask):
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        ### Progressive Refinement Module in MGMatting Paper
        alpha_pred = alpha_pred_os8
        
        # OS8 -> OS4
        weight_os4 = compute_unknown(alpha_pred, k_size=27, is_train=self.training) * detail_mask
        weight_os4 = (weight_os4 > 0).type(alpha_pred.dtype)
        alpha_pred_os4 = alpha_pred_os4.type(alpha_pred.dtype)
        alpha_pred = alpha_pred_os4 * weight_os4 + alpha_pred * (1 - weight_os4)

        # OS4 -> OS1
        weight_os1 = compute_unknown(alpha_pred, k_size=15, is_train=self.training) * detail_mask
        weight_os1 = (weight_os1 > 0).type(alpha_pred.dtype)
        alpha_pred_os1 = alpha_pred_os1.type(alpha_pred.dtype)
        alpha_pred = alpha_pred_os1 * weight_os1 + alpha_pred * (1 - weight_os1)

        return alpha_pred, weight_os4, weight_os1
    
    def forward(self, x, mid_fea, b, n_f, n_i, masks, iter, gt_alphas, **kwargs):
        '''
        masks: [b * n_f * n_i, 1, H, W]
        '''
        x, masks, valid_masks, gt_masks, fea1, fea2, fea3, image, h, w = self.os32_to_os8(x, mid_fea, b, n_f, n_i, masks, gt_alphas)

        # Predict OS8
        # use mask attention during warmup of training
        use_mask_atten = iter < self.warmup_mask_atten_iter and self.training

        x_os8, x, queries, loss_max_atten, _ = self.refine_OS8(x, masks, use_mask_atten=use_mask_atten, gt_mask=gt_masks)
        x_os8 = F.interpolate(x_os8, size=(h, w), mode='bilinear', align_corners=False)
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

        if self.training:
            x_os8 = x_os8 * valid_masks
        else:
            x_os8 = x_os8[:, :n_i]
        
        # Warm-up - Using gt_alphas instead of x_os8 for later steps
        guided_mask_os8 = x_os8.clone()
        is_use_alphas_gt = False
        if self.training and (iter < self.warmup_detail_iter or x_os8.sum() == 0 or (iter < self.warmup_detail_iter * 3 and random.random() < 0.5)):
            guided_mask_os8 = gt_alphas.clone()
            is_use_alphas_gt = True
        
        unknown_os8 = compute_unknown(guided_mask_os8, k_size=30)

        # Dummy code to prevent all zeros
        x_os4, x_os1 = self.process_os4_os1(x, b, n_f, fea1, fea2, fea3, image, x_os8, queries, guided_mask_os8, unknown_os8)
        
        # Prepare outpur
        ret = {}
        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8

        alpha_pred, weight_os4, weight_os1 = self.fuse(ret, unknown_os8)

        ret['refined_masks'] = alpha_pred
        
        # If we use GT alphas, select weight randomly in the unknown mask region (for loss computation later)
        if is_use_alphas_gt:
            weight_os4 = compute_unknown(gt_alphas, k_size=30, is_train=self.training) * unknown_os8
            weight_os1 = compute_unknown(gt_alphas, k_size=15, is_train=self.training) * unknown_os8
        
        ret['weight_os4'] = weight_os4
        ret['weight_os1'] = weight_os1 
        ret['detail_mask'] = unknown_os8 # Mask of refined points

        if self.training and iter >= self.warmup_mask_atten_iter:
            ret['loss_max_atten'] = loss_max_atten
        return ret

    def process_os4_os1(self, x, b, n_f, fea1, fea2, fea3, image, x_os8, queries, guided_mask_os8, unknown_os8):
        if unknown_os8.max() == 0 and self.training:
            unknown_os8[:, :, 200: 250, 200: 250] = 1.0

        if unknown_os8.sum() > 0 or self.training:
            # Expand queries to n_f
            queries = queries[:, None].expand(-1, n_f, -1, -1).reshape(b * n_f, *queries.shape[1:])
            
            # Predict details at OS4 and OS1
            x_os4, x_os1, mem_details = self.predict_details(x, image, unknown_os8, guided_mask_os8, mem_details, queries, [fea1, fea2, fea3])

            x_os4 = x_os4.reshape(b * n_f, guided_mask_os8.shape[1], *x_os4.shape[-2:])
            x_os1 = x_os1.reshape(b * n_f, guided_mask_os8.shape[1], *x_os1.shape[-2:])

            x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
            x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
            x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        else:
            x_os4 = torch.zeros((b * n_f, x_os8.shape[1], image.shape[2], image.shape[3]), device=x_os8.device)
            x_os1 = torch.zeros_like(x_os4)
        return x_os4,x_os1

    def os32_to_os8(self, x, mid_fea, b, n_f, n_i, masks, gt_alphas):
        masks = masks.reshape(b, n_f, n_i, masks.shape[2], masks.shape[3])
        valid_masks = masks.flatten(0,1).sum((2, 3), keepdim=True) > 0
        gt_masks = None
        
        # In training, use gt_alphas to supervise the attention
        if self.training:
            gt_masks = (gt_alphas > 0).reshape(b, n_f, n_i, gt_alphas.shape[2], gt_alphas.shape[3])

            if gt_masks.shape[-1] != masks.shape[-1]:
                gt_masks = resizeAnyShape(gt_masks, scale_factor=masks.shape[-1] * 1.0/ gt_masks.shape[-1], use_max_pool=True)

        # OS32 -> OS 8
        
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        
        image = mid_fea['image']
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        h, w = image.shape[-2:]
        return x,masks,valid_masks,gt_masks,fea1,fea2,fea3,image,h,w

def res_shortcut_inst_matt_spconv_22(**kwargs):
    return ResShortCut_InstMattSpconv_Dec(BasicBlock, [2, 3, 3, 2], **kwargs)