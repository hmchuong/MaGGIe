from functools import partial
import logging
import torch
import random
from torch import nn
from torch.nn import functional as F
import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp
from vm2m.network.ops import SpectralNorm
from vm2m.network.module.base import conv1x1
from vm2m.network.module.mask_matte_embed_atten import MaskMatteEmbAttenHead
from vm2m.network.module.instance_matte_head import InstanceMatteHead
from vm2m.network.module.temporal_nn import TemporalNN
from vm2m.network.module.ligru_conv import LiGRUConv
from vm2m.network.module.stm import STM

from .resnet_dec import BasicBlock

class ResShortCut_AttenSpconv_Dec(nn.Module):
    def __init__(self, block, layers, norm_layer=None, large_kernel=False, 
                 late_downsample=False, final_channel=32,
                 atten_dim=128, atten_block=2, 
                 atten_head=1, atten_stride=1, head_channel=32, max_inst=10, use_atten_mask=True, **kwargs):
        super(ResShortCut_AttenSpconv_Dec, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.large_kernel = large_kernel
        self.kernel_size = 5 if self.large_kernel else 3

        self.inplanes = 512 if layers[0] > 0 else 256
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32

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
            use_mask_atten=use_atten_mask
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
    
    def predict_details(self, x, image, roi_masks, masks):
        '''
        x: [b, 64, h/4, w/4], os4 semantic features
        image: [b, 3, H, W], original image
        masks: [b, n_i, H, W], dilated guided masks from OS8 prediction
        '''
        # Stack masks and images
        b, n_i, H, W = masks.shape
        masks = masks.reshape(b * n_i, 1, H, W)
        roi_masks = roi_masks.reshape(b * n_i, 1, H, W)
        image = image.unsqueeze(1).repeat(1, n_i, 1, 1, 1).reshape(b * n_i, 3, H, W)
        inp = torch.cat([image, masks], dim=1)

        # Prepare sparse tensor
        coords = torch.where(roi_masks.squeeze(1) > 0)
        # if self.training and len(coords[0]) > 1600000:
        #     ids = torch.randperm(len(coords[0]))[:1600000]
        #     coords = [i[ids] for i in coords]
        
        inp = inp.permute(0, 2, 3, 1).contiguous()
        inp = inp[coords]
        coords = torch.stack(coords, dim=1)
        # print(coords.shape[0])
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
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x[coords[:, 0], coords[:, 1], coords[:, 2]]
        x = spconv.SparseConvTensor(x, fea3.indices, (H // 4, W // 4), b * n_i, indice_dict=fea3.indice_dict)
        x = Fsp.sparse_add(x, fea3)

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
        
        return x_os4_out, x_os1_out
    
    def new_predict_details(self, x, inp, roi_masks, masks):
        '''
        x: [b, 64, h/4, w/4], os4 semantic features
        image: [b, 3, H, W], original image
        masks: [b, n_i, H, W], dilated guided masks from OS8 prediction
        '''
        

        # Stack masks and images
        b, n_i, H, W = masks.shape
        masks = masks.reshape(b * n_i, 1, H, W)
        roi_masks = roi_masks.reshape(b * n_i, 1, H, W)
        inp = inp.unsqueeze(1).repeat(1, n_i, 1, 1, 1).reshape(b * n_i, 3, H, W)
        inp = torch.cat([inp, masks], dim=1)

        # Prepare sparse tensor
        coords = torch.where(roi_masks.squeeze(1) > 0)
        # if self.training and len(coords[0]) > 1600000:
        #     ids = torch.randperm(len(coords[0]))[:1600000]
        #     coords = [i[ids] for i in coords]
        
        inp = inp.permute(0, 2, 3, 1).contiguous()
        inp = inp[coords]
        coords = torch.stack(coords, dim=1)
        # print(coords.shape[0])
        inp = spconv.SparseConvTensor(inp, coords.int(), (H, W), b * n_i)

        # import pdb; pdb.set_trace()

        # inp -> OS 1 --> OS 2 --> OS 4
        inp = self.low_os1_module(inp)
        fea2 = self.low_os2_module(inp)
        fea3 = self.low_os4_module(fea2)

        # Combine x with fea3
        # Prepare sparse tensor of x
        coords = fea3.indices.clone()
        coords[:, 0] = torch.div(coords[:, 0], n_i, rounding_mode='floor')
        coords = coords.long()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x[coords[:, 0], coords[:, 1], coords[:, 2]]
        x = spconv.SparseConvTensor(x, fea3.indices, (H // 4, W // 4), b * n_i, indice_dict=fea3.indice_dict)
        x = Fsp.sparse_add(x, fea3)

        # Predict OS 4
        x_os4 = self.refine_OS4(x)
        coords = x_os4.indices.long()
        x_os4 = x_os4.dense()
        x_os4 = x_os4 - 99
        x_os4[coords[:, 0], :, coords[:, 1], coords[:, 2]] += 99

        # Combine with fea2
        # - Upsample to OS 2
        x = self.layer4(x)
        x = Fsp.sparse_add(x, fea2)

        # Combine with fea1
        # - Upsample to OS 1
        x = self.layer5(x)
        x = Fsp.sparse_add(x, inp)

        # Predict OS 1
        x = self.refine_OS1(x)
        coords = x.indices.long()
        x = x.dense()
        x = x - 99
        x[coords[:, 0], :, coords[:, 1], coords[:, 2]] += 99
        
        return x_os4, x

    def compute_unknown(self, masks, k_size=30):
        h, w = masks.shape[-2:]
        uncertain = (masks > 1.0/255.0) & (masks < 254.0/255.0)
        dilated_m = F.max_pool2d(uncertain.float(), kernel_size=k_size, stride=1, padding=k_size // 2)
        dilated_m = dilated_m[:,:, :h, :w]
        return dilated_m

    def forward(self, x, mid_fea, b, n_f, n_i, masks, iter, logit_output=False, temp_aggre_func=None, **kwargs):
        '''
        masks: [b * n_f * n_i, 1, H, W]
        '''

        # Reshape masks
        masks = masks.reshape(b, n_f, n_i, masks.shape[2], masks.shape[3])
        valid_masks = masks.flatten(0,1).sum((2, 3), keepdim=True) > 0

        # OS32 -> OS 8
        ret = {}
        fea4, fea5 = mid_fea['shortcut']
        
        image = mid_fea['image']
        x = self.layer1(x) + fea5
        
        if temp_aggre_func is not None:
            x = temp_aggre_func(x)
            if not self.training:
                ret['embedding_os8'] = x

        x = self.layer2(x) + fea4

        # Predict OS8
        x_os8, x, loss_max_atten, loss_min_atten = self.refine_OS8(x, masks)
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0
        if self.training:
            x_os8 = x_os8 * valid_masks
        else:
            # x_os8[:, n_i:] = 0.0
            x_os8 = x_os8[:, :n_i]

        x = self.layer3(x)

        # Start from here
        unknown_os8 = self.compute_unknown(x_os8)
        if unknown_os8.sum() > 0:
            x_os4, x_os1 = self.predict_details(x, image, unknown_os8, x_os8)
            x_os4 = x_os4.reshape(b * n_f, x_os8.shape[1], *x_os4.shape[-2:])
            x_os1 = x_os1.reshape(b * n_f, x_os8.shape[1], *x_os1.shape[-2:])
            
            if not logit_output:
                x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
                x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
                x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        else:
            x_os4 = torch.zeros((b * n_f, x_os8.shape[1], image.shape[2], image.shape[3]), device=x_os8.device)
            x_os1 = torch.zeros_like(x_os4)
        

        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8
        ret['loss_max_atten'] = loss_max_atten
        ret['loss_min_atten'] = loss_min_atten
        return ret

class ResShortCut_AttenSpconv_Temp_Dec(ResShortCut_AttenSpconv_Dec):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.temp_module = TemporalNN(512)
        # self.temp_module_os8 = LiGRUConv(128)
        # self.temp_module_o232 = STM(512, os=32, embedd_dim=kwargs["embed_dim"])
        self.temp_module_os8 = STM(256, os=16, embedd_dim=kwargs["embed_dim"])
        # self.detail_temp_os4 = nn.Sequential(
        #     nn.PixelUnshuffle(2),
        #     nn.Conv2d(8, 4, 3, padding=1, groups=4),
        #     nn.PixelShuffle(2)
        # )
        # self.detail_temp_os1 = nn.Sequential(
        #     nn.PixelUnshuffle(8),
        #     nn.Conv2d(128, 64, 3, padding=1, groups=64),
        #     nn.PixelShuffle(8)
        # )

        # for module in [self.detail_temp_os1, self.detail_temp_os4]:
        #     for m in module.modules():
        #         if isinstance(m, nn.Conv2d):
        #             nn.init.xavier_uniform_(m.weight)
        #             m.bias.data.zero_()

    def aggregate_detail_matte(self, temp_module, x, b, n_f):
        x = x.reshape(b, n_f, -1, *x.shape[-2:])
        new_x = [x[:, 0]]
        
        for i in range(1, n_f):
            cur_f = x[:, i].flatten(0, 1)
            prev_f = new_x[-1].flatten(0, 1)
            inp = torch.stack([prev_f, cur_f], dim=1)
            cur_f = temp_module(inp)
            cur_f = cur_f.reshape(b, -1, *cur_f.shape[-2:])
            new_x.append(cur_f)
        new_x = torch.stack(new_x, dim=1)
        new_x = new_x.flatten(0, 1)
        return new_x

    def temp_aggregation(self, x, temp_module, b, n_f, prev_feat=None, prev_mask=None):
        if self.training or n_f > 1  or prev_feat is not None:
            # print("Perform embedding aggregation")
            # import pdb; pdb.set_trace()
            if prev_feat is not None:
                x = torch.cat((prev_feat, x.unsqueeze(1)), dim=1)
                x = x.flatten(0, 1)
            # import pdb; pdb.set_trace()
            x = x.reshape(b, -1, *x.shape[1:])
            if not self.training:
                x = temp_module(x, prev_mask=prev_mask)
            else:
                new_x = []
                for i in range(1, n_f):
                    new_x.append(temp_module(x[:, :i+1], prev_mask=None)[:, -1])
                new_x = [temp_module(torch.flip(x, dims=(1,)), prev_mask=None)[:, -1]] + new_x
                x = torch.stack(new_x, dim=1)

            x = x.flatten(0, 1)
            
            if prev_feat is not None:
                x =  x.reshape(b, -1, *x.shape[-3:])
                x = x[:, -1, :, :]
        return x

    def forward(self, x, mid_fea, b, n_f, n_i, masks, iter, prev_feat, prev_mask, **kwargs):
        
        # Perform temporal aggregation
        # if prev_feat is None:
        #     prev_feat = {}
        
        # with torch.no_grad():
        # x = self.temp_aggregation(x, self.temp_module_o232, b, n_f, prev_feat=prev_feat.get('os32', None), prev_mask=prev_mask)
        # output = super().forward(x, mid_fea, b, n_f, n_i, masks, iter, logit_output=False, **kwargs)
        output = super().forward(x, mid_fea, b, n_f, n_i, masks, iter, logit_output=False, 
                                 temp_aggre_func=partial(self.temp_aggregation, 
                                                         temp_module=self.temp_module_os8,  
                                                         b=b, n_f=n_f, prev_feat=prev_feat.get('os8', None), 
                                                         prev_mask=prev_mask), 
                                 **kwargs)
        # if not self.training:
        #     output['embedding_os32'] = x
        # if prev_mask is not None or n_f > 1:
        #     # print("Perform detail aggregation")
        #     # Propagate the matte between frames
        #     x_os4 = output['alpha_os4']
        #     x_os1 = output['alpha_os1']
        #     if prev_mask is not None:
        #         x_os4 = torch.cat([prev_mask[0], x_os4], dim=0)
        #         x_os1 = torch.cat([prev_mask[1], x_os1], dim=0)
        #         n_f  += 1
        #     x_os4 = self.aggregate_detail_matte(self.detail_temp_os4, x_os4, b, n_f)
        #     x_os1 = self.aggregate_detail_matte(self.detail_temp_os1, x_os1, b, n_f)
        #     if prev_mask is not None:
        #         x_os4 = x_os4[:-1]
        #         x_os1 = x_os1[:-1]
        #     output['alpha_os4'] = x_os4
        #     output['alpha_os1'] = x_os1
        
        # if not self.training:
        #     output['prev_mask'] = (output['alpha_os4'], output['alpha_os1'])

        # output['alpha_os4'] = F.interpolate(output['alpha_os4'], scale_factor=4.0, mode='bilinear', align_corners=False)
        # output['alpha_os4'] = (torch.tanh(output['alpha_os4']) + 1.0) / 2.0
        # output['alpha_os1'] = (torch.tanh(output['alpha_os1']) + 1.0) / 2.0
        # Temporal consistency with previous frame
        return output

def res_shortcut_attention_spconv_decoder_22(**kwargs):
    return ResShortCut_AttenSpconv_Dec(BasicBlock, [2, 3, 3, 2], **kwargs)

def res_shortcut_attention_spconv_temp_decoder_22(**kwargs):
    return ResShortCut_AttenSpconv_Temp_Dec(block=BasicBlock, layers=[2, 3, 3, 2], **kwargs)