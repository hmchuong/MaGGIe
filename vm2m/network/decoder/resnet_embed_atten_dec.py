import logging
import torch
import random
from torch import nn
from torch.nn import functional as F
from vm2m.network.ops import SpectralNorm
from .resnet_dec import BasicBlock
from vm2m.network.module.base import conv1x1
from vm2m.network.module.mask_matte_embed_atten import MaskMatteEmbAttenHead
from vm2m.network.module.instance_matte_head import InstanceMatteHead
from vm2m.network.module.temporal_nn import TemporalNN

class ResShortCut_EmbedAtten_Dec(nn.Module):
    def __init__(self, block, layers, norm_layer=None, large_kernel=False, 
                 late_downsample=False, final_channel=32,
                 atten_dim=128, atten_block=2, 
                 atten_head=1, atten_stride=1, head_channel=32, max_inst=10, **kwargs):
        super(ResShortCut_EmbedAtten_Dec, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.large_kernel = large_kernel
        self.kernel_size = 5 if self.large_kernel else 3

        self.inplanes = 512 if layers[0] > 0 else 256
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32

        self.conv1 = SpectralNorm(nn.ConvTranspose2d(self.midplanes, 32, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn1 = norm_layer(32)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.tanh = nn.Tanh()
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.midplanes, layers[3], stride=2)

        ## OS1
        # self.refine_OS1 = MaskMatteEmbAttenHead(
        #     input_dim=32,
        #     atten_stride=atten_strides[0],
        #     attention_dim=atten_dims[0],
        #     n_block=atten_blocks[0],
        #     n_head=atten_heads[0],
        #     output_dim=final_channel,
        #     return_feat=False,
        #     max_inst=max_inst,
        # )
        
        ## OS4
        # self.refine_OS4 = MaskMatteEmbAttenHead(
        #     input_dim=64,
        #     atten_stride=atten_strides[1],
        #     attention_dim=atten_dims[1],
        #     n_block=atten_blocks[1],
        #     n_head=atten_heads[1],
        #     output_dim=final_channel,
        #     max_inst=max_inst,
        #     return_feat=False,
        # )
        
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
            use_temp_pe=False
        )

        group = max_inst if head_channel % max_inst == 0 else 1
        self.use_sep_head = False
        if self.use_sep_head:
            head_channel = 32
            ## 1 scale
            self.refine_OS1 = nn.Sequential(
                nn.Conv2d(32 + 1, head_channel, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
                # norm_layer(head_channel),
                nn.GroupNorm(4, head_channel),
                self.leaky_relu,
                # nn.Conv2d(head_channel, max_inst, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, groups=group)
                nn.Conv2d(head_channel, 1, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
            )
                ## 1/4 scale
            self.refine_OS4 = nn.Sequential(
                nn.Conv2d(64 + 1, head_channel, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
                # norm_layer(head_channel),
                nn.GroupNorm(4, head_channel),
                self.leaky_relu,
                nn.Conv2d(head_channel, 1, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
            )
        else:
            self.refine_OS1 = nn.Sequential(
                nn.Conv2d(32, head_channel, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
                norm_layer(head_channel),
                self.leaky_relu,
                nn.Conv2d(head_channel, max_inst, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, groups=group)
            )
            # self.refine_OS1 = InstanceMatteHead(self.refine_OS8.query_feat, 32, hidden_channel=32, k_out_channel=8)
        
            ## 1/4 scale
            self.refine_OS4 = nn.Sequential(
                nn.Conv2d(64, head_channel, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False),
                norm_layer(head_channel),
                self.leaky_relu,
                nn.Conv2d(head_channel, max_inst, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, groups=group)
            )
        # self.refine_OS4 = InstanceMatteHead(self.refine_OS8.query_feat, 64, hidden_channel=32, k_out_channel=8)
        
        
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

    def compute_next_input_mask(self, input_masks, prev_masks, iter):
        # warmup = 5000

        # if not self.training:
        #     import pdb; pdb.set_trace()
        # prev_masks = prev_masks.reshape(*input_masks.shape[:-2], *prev_masks.shape[-2:])
        # if self.training and (iter < warmup or (iter < warmup * 3 and random.randint(0,1) == 0)):
        #     # Use masks from the input
        #     prev_masks = input_masks
        
        # return prev_masks
        return input_masks

    def predict_detail(self, head, x, input_masks, prev_masks):
        '''
        x: b*n_f x c x h x w, detailed feature
        prev_masks: b*n_f x n_i x h x w, previous masks
        '''
        padding_size = int(30 / prev_masks.shape[-1] * x.shape[-1])
        divisible_size = int(32 / prev_masks.shape[-1] * x.shape[-1])

        # scale prev_masks to match x
        prev_masks = F.interpolate(prev_masks, size=x.shape[-2:], mode='bilinear', align_corners=False)
        input_masks = F.interpolate(input_masks, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # for each mask, get coordinates, crop and extend to be divisible by 32, then predict
        logging.debug(f"prev_masks: {prev_masks.shape} {prev_masks.sum(dim=[2,3])}")
        valid_mask = torch.nonzero(input_masks.sum(dim=[2,3]) > 1.0/255.0)
        output_logit = torch.full(size=(x.shape[0], prev_masks.shape[1], x.shape[2], x.shape[3]), fill_value=-5, device=x.device, dtype=x.dtype)
        logging.debug("Start predicting")
        for b_i, inst_i in valid_mask:
            feat = x[b_i]
            logging.debug(f"Processing: {b_i}, {inst_i}")
            coords = torch.nonzero(input_masks[b_i, inst_i] > 0.5)
            y_min, x_min = coords.min(dim=0)[0]
            y_max, x_max = coords.max(dim=0)[0]
            # extend to have padding_size
            x_min = max(0, x_min - padding_size)
            y_min = max(0, y_min - padding_size)
            x_max = min(x_max + padding_size, feat.shape[-1] - 1)
            y_max = min(y_max + padding_size, feat.shape[-2] - 1)
            
            # extend to have multiply by 32
            x_min = torch.div(x_min, divisible_size, rounding_mode='floor') * divisible_size
            y_min = torch.div(y_min, divisible_size, rounding_mode='floor') * divisible_size
            x_max = (torch.div(x_max, divisible_size, rounding_mode='floor') + 1) * divisible_size
            y_max = (torch.div(y_max, divisible_size, rounding_mode='floor') + 1) * divisible_size

            # crop feat
            feat = feat[:, y_min:y_max, x_min:x_max]
            mask = prev_masks[b_i, inst_i, y_min:y_max, x_min:x_max]

            # Stack mask and feat
            feat = torch.cat([feat, mask.unsqueeze(0)], dim=0)

            # predict
            pred = head(feat.unsqueeze(0))
            # paste back
            output_logit[b_i, inst_i, y_min:y_max, x_min:x_max] = output_logit[b_i, inst_i, y_min:y_max, x_min:x_max] * 0 + pred[0, 0]
        
            logging.debug("done processing")
        return output_logit


    def forward(self, x, mid_fea, b, n_f, n_i, masks, iter, **kwargs):
        '''
        masks: [b * n_f * n_i, 1, H, W]
        '''

        # Reshape masks
        masks = masks.reshape(b, n_f, n_i, masks.shape[2], masks.shape[3])
        valid_masks = masks.flatten(0,1).sum((2, 3), keepdim=True) > 0


        ret = {}
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        
        logging.debug("forwarding os8")
        # import pdb; pdb.set_trace() 
        x_os8, x = self.refine_OS8(x, masks)
        # x_os8 = self.refine_OS8(x, masks)
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0
        if self.training:
            x_os8 = x_os8 * valid_masks
        else:
            x_os8[:, n_i:] = 0.0

        x = self.layer3(x) + fea3

        # Compute new masks from the x_os8
        # prev_mask = self.compute_next_input_mask(masks, x_os8, iter)
        # x_os4, x = self.refine_OS4(x, prev_mask)
        # x_os4 = self.refine_OS4(x, prev_mask)
        if not self.use_sep_head:
            x_os4 = self.refine_OS4(x)
        else:
            logging.debug("forwarding os4")
            x_os4 = self.predict_detail(self.refine_OS4, x, masks.flatten(0,1), x_os8)
        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
        # x_os4 = x_os4 * valid_masks
        if self.training:
            x_os4 = x_os4 * valid_masks
        else:
            x_os4[:, n_i:] = 0.0

        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1

        # Compute new masks from the x_os4
        # prev_mask = self.compute_next_input_mask(masks, x_os4, iter)
        # x_os1 = self.refine_OS1(x, prev_mask)
        if not self.use_sep_head:
            x_os1 = self.refine_OS1(x)
        else:
            logging.debug("forwarding os1")
            x_os1 = self.predict_detail(self.refine_OS1, x, masks.flatten(0, 1), x_os8)
        x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0

        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8

        return ret

class ResShortCut_TempEmbedAtten_Dec(ResShortCut_EmbedAtten_Dec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temp_module = TemporalNN(512)
    def forward(self, x, mid_fea, b, n_f, n_i, masks, iter, **kwargs):
        
        # Perform temporal aggregation
        x = x.reshape(b, 3, *x.shape[1:])
        x = self.temp_module(x)
        x = x.reshape(b * 3, -1, *x.shape[-2:])

        return super().forward(x, mid_fea, b, n_f, n_i, masks, iter, **kwargs)
        

class ResS_EmbedAttenProdMask_Dec(ResShortCut_EmbedAtten_Dec):
    def compute_unknown(self, masks, k_size):
        masks = masks.clone()
        masks[masks < 1.0/255.0] = 0.0
        masks[masks > 1 - 1.0/255.0] = 0.0
        dilated_m = F.max_pool2d(masks, kernel_size=k_size, stride=1, padding=k_size // 2)
        # erosion_m = F.max_pool2d(1 - masks, kernel_size=k_size, stride=1, padding=k_size // 2)
        dilated_m = dilated_m[:, :, :masks.shape[2], :masks.shape[3]]
        # import pdb; pdb.set_trace()
        # unk_m = dilated_m - erosion_m
        # unk_m = unk_m[:, :, :masks.shape[2], :masks.shape[3]]
        return dilated_m #unk_mdilated
    
    def forward(self, x, mid_fea, b, n_f, n_i, masks, iter, warmup_iter, gt_alphas, **kwargs):
        '''
        masks: [b * n_f * n_i, 1, H, W]
        '''

        # Reshape masks
        masks = masks.reshape(b, n_f, n_i, masks.shape[2], masks.shape[3])

        ret = {}
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        
        # import pdb; pdb.set_trace() 
        # x_os8, x = self.refine_OS8(x, masks)
        x_os8 = self.refine_OS8(x, masks)
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode='bilinear', align_corners=False)
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

        x = self.layer3(x) + fea3

        # Compute new masks from the x_os8
        prev_mask = x_os8
        if self.training and (iter < warmup_iter or (iter < warmup_iter * 3 and random.randint(0,1) == 0)):
            # Use masks from the input
            prev_mask = gt_alphas.reshape(b * n_f, n_i, *gt_alphas.shape[-2:])
            prev_mask = (prev_mask > 0.0).float()
        if prev_mask.shape[1] < x_os8.shape[1]:
            prev_mask = torch.cat([prev_mask, torch.zeros_like(x_os8[:, prev_mask.shape[1]:])], dim=1)

        # import pdb; pdb.set_trace()
        # os4_masks = self.compute_unknown(prev_mask, k_size=30)
        os4_masks = prev_mask
        os4_masks = os4_masks.reshape(b, n_f, -1, *os4_masks.shape[-2:])

        x_os4 = self.refine_OS4(x, os4_masks)
        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode='bilinear', align_corners=False)
        x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0

        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1

        # Compute new masks from the x_os8
        prev_mask = x_os4
        if self.training and (iter < warmup_iter or (iter < warmup_iter * 3 and random.randint(0,1) == 0)):
            # Use masks from the input
            prev_mask = gt_alphas.reshape(b * n_f, n_i, *gt_alphas.shape[-2:])
            prev_mask = (prev_mask > 0.0).float()
        else:
            # OS8 masks + OS4 masks
            uncertainty = x_os8.clone()
            prev_mask = x_os8.clone()
            uncertainty[prev_mask < 1.0/255.0] = 0.0
            uncertainty[prev_mask > 1 - 1.0/255.0] = 0.0
            uncertainty = F.max_pool2d(uncertainty, kernel_size=15, stride=1, padding=15 // 2)
            prev_mask[uncertainty > 0.0] = x_os4[uncertainty > 0.0]

        if prev_mask.shape[1] < x_os4.shape[1]:
            prev_mask = torch.cat([prev_mask, torch.zeros_like(x_os4[:, prev_mask.shape[1]:])], dim=1)

        os1_masks = prev_mask
        # os1_masks = self.compute_unknown(prev_mask, k_size=15)
        os1_masks = os1_masks.reshape(b, n_f, -1, *os1_masks.shape[-2:])

        x_os1 = self.refine_OS1(x, os1_masks)
        x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0

        ret['alpha_os1'] = x_os1
        ret['alpha_os4'] = x_os4
        ret['alpha_os8'] = x_os8

        return ret

def res_shortcut_embed_attention_decoder_22(**kwargs):
    return ResShortCut_EmbedAtten_Dec(BasicBlock, [2, 3, 3, 2], **kwargs)

def res_shortcut_embed_attention_proma_decoder_22(**kwargs):
    return ResS_EmbedAttenProdMask_Dec(BasicBlock, [2, 3, 3, 2], **kwargs)

def res_shortcut_temp_embed_atten_decoder_22(**kwargs):
    return ResShortCut_TempEmbedAtten_Dec(block=BasicBlock, layers=[2, 3, 3, 2], **kwargs)