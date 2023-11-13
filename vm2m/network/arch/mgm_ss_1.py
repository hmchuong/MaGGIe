import cv2
import random
import numpy as np
import torch
from torch.nn import functional as F
import albumentations as A
import imgaug.augmenters as iaa
from imgaug import parameters as iap
from yacs.config import CfgNode as CN
from vm2m.network.loss import loss_dtSSD
from .mgm_tempspar import MGM_TempSpar

class MGM_SS(MGM_TempSpar):
    def __init__(self, backbone, decoder, cfg):
        super().__init__(backbone, decoder, cfg)
        
        # Augmentation: flip, gamma contrast, motion blur, AdditiveGaussianNoise, JpegCompression
        self.motion_aug = A.MotionBlur(p=1.0, blur_limit=(3, 49)) # for image, mask. alpha
        self.pixel_aug_gamm = iaa.GammaContrast(gamma=iap.TruncatedNormal(1.0, 0.2, 0.5, 1.5)) # for image
        self.pixel_aug_gaussian = iaa.AdditiveGaussianNoise(scale=(0, 0.03*255)) # for image
        self.jpeg_aug = iaa.JpegCompression(compression=(60, 90)) # for image

        kernel_size = 30
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


        # Load image model
        img_config = "output/HIM/ours_1110_stronger-aug_guidance_scratch/config.yaml"
        img_weights = "output/HIM/ours_1110_stronger-aug_guidance_scratch/last_model_24k.pth"
        img_config = CN().load_cfg(open(img_config, "r"))
        
        from vm2m.network import build_model
        img_model = build_model(img_config.model)
        img_model.load_state_dict(torch.load(img_weights))
        self.image_model = img_model
        self.train()
    
    def train(self, mode=True):
        super().train(mode)
        self.image_model.eval()
        for p in self.image_model.parameters():
            p.requires_grad = False
    
    def state_dict(self):
        state_dict = super().state_dict()
        return_state_dict = {}
        for k in state_dict.keys():
            if k.startswith("image_model"):
                continue
            return_state_dict[k] = state_dict[k]
        return return_state_dict

    def generate_motion_input(self, images, masks, kernel):
        '''
        images: (b, n_f, 3, h, w)
        masks: (b, n_f, n_i, h, w)
        kernel: (k, k)
        '''
        b, n_f, _, h, w = images.shape
        if masks.shape[-1] != w:
            masks = masks.flatten(0,1)
            masks = F.interpolate(masks, size=(h, w), mode="nearest")
        
        # Build input: (b * n_f, 3 + n_i, h, w)
        inputs = torch.cat([images.flatten(0,1), masks.flatten(0,1)], dim=1)
        kernel = kernel.type_as(inputs)
        kernel = kernel.expand(inputs.shape[1], 1, -1, -1)
        
        # Apply motion blur
        out = F.conv2d(inputs, kernel, padding=kernel.shape[-1]//2, groups=inputs.shape[1])
        out_image = out[:, :3]
        out_image = out_image.view(b, n_f, -1, h, w)
        out_masks = out[:, 3:]
        out_masks = out_masks.view(b, n_f, -1, h, w)
        out_masks = (out_masks > 0).type_as(out_masks)

        return out_image, out_masks
    
    def drop_mask(self, masks):
        '''
        masks: (b, n_f, n_i, h, w)
        '''
        b, n_f, n_i, h, w = masks.shape
        valid_masks = masks.sum((-1, -2)) > 0
        valid_mask_ids = torch.nonzero(valid_masks, as_tuple=True)
        if len(valid_mask_ids[0]) // 2 <= 1:
            return masks
        n_dropouts = np.random.randint(1, len(valid_mask_ids[0]) // 2)
        selected_indices = np.random.choice(len(valid_mask_ids[0]), n_dropouts, replace=False)
        for i in selected_indices:
            b_id = valid_mask_ids[0][i]
            f_id = valid_mask_ids[1][i]
            i_id = valid_mask_ids[2][i]
            mask = masks[b_id, f_id, i_id].detach().cpu().numpy()
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                continue
            xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
            if (ymax - ymin + 1) // 8 < 2 or (xmax - xmin + 1) // 8 < 2:
                continue
            
            perturb_size_h, perturb_size_w = np.random.randint((ymax - ymin + 1) // 8, (ymax - ymin + 1) // 4), np.random.randint((xmax - xmin + 1) // 8, (xmax - xmin + 1) // 4)
            idx = np.random.choice(range(len(ys)), 1)
            x, y = int(xs[idx]), int(ys[idx])
            
            x = min(x, xmax - perturb_size_w)
            y = min(y, ymax - perturb_size_h)
            # print("Drop mask: ", b_id, f_id, i_id, x, y, perturb_size_w, perturb_size_h)
            # cv2.imwrite("pred.png", masks[b_id, f_id, i_id].cpu().numpy() * 255)
            # import pdb; pdb.set_trace()
            masks[b_id, f_id, i_id, y:y+perturb_size_h, x:x+perturb_size_w] = 0
            # cv2.imwrite("pred.png", masks[b_id, f_id, i_id].cpu().numpy() * 255)
            # import pdb; pdb.set_trace()
        return masks
    
    def generate_aug_input(self, images, masks, motion_kernel):
        '''
        images: (b, n_f, 3, h, w)
        masks: (b, n_f, n_i, h, w)
        '''
        # Flip
        flip = random.random() > 0.5
        if flip:
            images = images.flip(-1)
            masks = masks.flip(-1)

        # Motion aug
        images, masks = self.generate_motion_input(images, masks, motion_kernel)

        # Denormalize image
        image_dtype = images.dtype
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).type_as(images)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).type_as(images)
        images = images * std + mean
        images = images.clamp(0, 1).permute(0, 1, 3, 4, 2)
        images = (images * 255).cpu().numpy().astype('uint8')
        
        
        # Gamma contrast aug
        gamma_aug = self.pixel_aug_gamm.to_deterministic()
        noise_aug = self.pixel_aug_gaussian.to_deterministic()
        jpeg_aug = self.jpeg_aug.to_deterministic()
        is_gamma = random.random() < 0.2
        is_noise = random.random() < 0.2
        is_jpeg = random.random() < 0.2    
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                img = images[i, j]
                if is_gamma:
                    img = gamma_aug.augment_image(img)
                if is_noise:
                    img = noise_aug.augment_image(img)
                if is_jpeg:
                    img = jpeg_aug.augment_image(img)
                images[i, j] = img
        
        # Normalize image
        images = images.astype('float32') / 255
        images = torch.from_numpy(images).permute(0, 1, 4, 2, 3).type(image_dtype).to(masks.device)
        images = (images - mean) / std
        
        # Augment masks - random drop some regions in masks
        masks = self.drop_mask(masks)
        
        return images, masks, flip
        

    def extract_batch(self, batch, select_mask):
        new_batch = {}
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                new_batch[k] = batch[k][select_mask]
        return new_batch
    
    def flatten_batch(self, batch):
        new_batch = {}
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor) and batch[k].dim() > 2:
                new_batch[k] = batch[k].flatten(0,1)[:, None]
        return new_batch
    
    def prepare_real_batch(self, batch, new_images, new_masks):
        batch["image"] = torch.cat([batch["image"], new_images], dim=0)
        batch["mask"] = torch.cat([batch["mask"], new_masks], dim=0)
        batch["alpha"] = torch.cat([batch["alpha"], torch.zeros_like(batch["alpha"])], dim=0)
        # batch["weight"] = torch.cat([batch["weight"], torch.zeros_like(batch["weight"])], dim=0)
        batch["transition"] = torch.cat([batch["transition"], torch.zeros_like(batch["transition"])], dim=0)

    def warp_alpha(self, detail_out, out, kernel, is_flip):
        # Combine OS8, OS4, and OS1: (b, 1, n_i, h, w)
        alpha_os1 = out['alpha_os1']
        alpha_os4 = out['alpha_os4']
        alpha_os8 = out['alpha_os8']
        refined_masks = out['refined_masks']
        n_samples = alpha_os1.shape[0]

        ori_alpha_os1 = alpha_os1[:n_samples//2]
        ori_alpha_os4 = alpha_os4[:n_samples//2]
        ori_alpha_os8 = alpha_os8[:n_samples//2]
        ori_refined_masks = refined_masks[:n_samples//2]
        
        detail_alpha_os1 = detail_out['alpha_os1']
        detail_alpha_os4 = detail_out['alpha_os4']
        detail_alpha_os8 = detail_out['alpha_os8']
        detail_refined_masks = detail_out['refined_masks']

        # Flip the output
        if is_flip:
            ori_alpha_os1 = ori_alpha_os1.flip(-1)
            ori_alpha_os4 = ori_alpha_os4.flip(-1)
            ori_alpha_os8 = ori_alpha_os8.flip(-1)
            ori_refined_masks = ori_refined_masks.flip(-1)
            detail_alpha_os1 = detail_alpha_os1.flip(-1)
            detail_alpha_os4 = detail_alpha_os4.flip(-1)
            detail_alpha_os8 = detail_alpha_os8.flip(-1)
            detail_refined_masks = detail_refined_masks.flip(-1)

        alpha_pred = {
            "alpha_os1": alpha_os1[n_samples//2:],
            "alpha_os4": alpha_os4[n_samples//2:],
            "alpha_os8": alpha_os8[n_samples//2:],
            "refined_masks": refined_masks[n_samples//2:],
        }

        # warp alpha
        b, n_f, n_i, h, w = alpha_os1.shape
        inputs = torch.cat([ori_alpha_os1, ori_alpha_os4, ori_alpha_os8, ori_refined_masks,
                            detail_alpha_os1, detail_alpha_os4, detail_alpha_os8, detail_refined_masks], dim=1)
        inputs = inputs.flatten(0,1)
        kernel = kernel.type_as(inputs)
        kernel = kernel.expand(inputs.shape[1], 1, -1, -1)
        
        # Apply motion blur
        out = F.conv2d(inputs, kernel, padding=kernel.shape[-1]//2, groups=inputs.shape[1])
        out = out.view(n_samples//2 , -1, n_i, h, w)
        alpha_pred_warp = {
            "alpha_os1": out[:, :n_f],
            "alpha_os4": out[:, n_f:2*n_f],
            "alpha_os8": out[:, 2*n_f:3*n_f],
            "refined_masks": out[:, 3*n_f:4*n_f]
        }
        detail_pred_warp = {
            "alpha_os1": out[:, 4*n_f:5*n_f],
            "alpha_os4": out[:, 5*n_f:6*n_f],
            "alpha_os8": out[:, 6*n_f:7*n_f],
            "refined_masks": out[:, 7*n_f:]
        }
        return detail_pred_warp, alpha_pred_warp, alpha_pred
    
    def merge_output(self, syn_out, real_out, real_mask):
        '''
        merge keys: alpha_os1, alpha_os4, alpha_os8, refined_masks
        '''
        merged_out = {}
        for k in ['alpha_os1', 'alpha_os4', 'alpha_os8', 'refined_masks']:
            b_syn, n_f, n_i, h, w = syn_out[k].shape
            real_val = real_out[k].reshape(-1, n_f, n_i, h, w)
            # TODO: it's not correct when all real data is used
            b_real = real_val.shape[0] // 2
            new_out = torch.zeros(b_syn + b_real, n_f, n_i, h, w).type_as(syn_out[k])
            new_out[~real_mask] = syn_out[k]
            new_out[real_mask] = real_val[:b_real]
            merged_out[k] = new_out
        return merged_out

    def dilate_mask(self, mask):
        mask_shape = mask.shape
        mask_device = mask.device
        mask_type = mask.dtype
        mask = mask.flatten(0, 2)
        mask = mask.permute(1, 2, 0)
        mask = mask.detach().cpu().numpy().astype('uint8')
        mask = cv2.dilate(mask, self.dilate_kernel)
        mask = torch.from_numpy(mask).to(mask_device).type(mask_type)
        mask = mask.permute(2, 0, 1)
        mask = mask.view(*mask_shape)
        return mask

    def forward(self, batch, is_real=False, **kwargs):
    
        if not is_real or not self.training:
            return super().forward(batch, is_real=is_real, **kwargs)
        
        iter = batch["iter"]
        valid_masks = batch['mask'].sum((-1, -2), keepdim=True) > 0
        
        # Apply motion blur to the input without gt and generate additional input
        motion_kernel = self.motion_aug.get_params()["kernel"]
        motion_kernel = torch.from_numpy(motion_kernel).to(batch["image"].device)

        aug_images, aug_masks, is_flip = self.generate_aug_input(batch["image"], batch["mask"], motion_kernel)
        # import pdb; pdb.set_trace()
        # For the input without gt, compute supervised loss for the input without gt
        # real_batch = self.extract_batch(batch, real_data_mask)
        real_batch = batch

        # Predict with image model
        with torch.no_grad():
            real_detail_out = self.image_model(real_batch)

        # Add motioned images and masks to the batch
        self.prepare_real_batch(real_batch, aug_images, aug_masks)
        
        real_batch["iter"] = iter
        
        # Predict with the video model
        real_out, _ = super().forward(real_batch, is_real=True, **kwargs)
        
        # Compute motioned alpha with motion kernel
        alpha_detail_warp, alpha_pred_warp, alpha_pred = self.warp_alpha(real_detail_out, real_out, motion_kernel, is_flip)

        detail_pred_warp = alpha_detail_warp["refined_masks"].detach()
        pred_warp = alpha_pred_warp["refined_masks"]
        pred_warp_os8 = alpha_pred_warp["alpha_os8"]
        pred = alpha_pred["refined_masks"]
        pred_os8 = alpha_pred["alpha_os8"]

        # FG/ Unknown mask: Dilate the input masks and union them
        dilated_masks = self.dilate_mask(real_batch['mask'][pred.shape[0]:])

        # Compute mask for BG, FG loss
        bg_masks = (dilated_masks < 0.5).float()
        bg_masks = bg_masks * valid_masks
        # remove uncertainty region predicted by the image model
        bg_masks = bg_masks * (detail_pred_warp < 0.1)

        fg_masks = (dilated_masks >= 0.5).float()
        fg_masks = fg_masks * valid_masks

        loss_dict = {}
        # BG loss outside the mask region to remove noise
        loss_dict['loss_bg_warp'] = (self.regression_loss(pred_warp, torch.zeros_like(pred_warp), weight=bg_masks) + self.regression_loss(pred_warp_os8, torch.zeros_like(pred_warp_os8), weight=bg_masks)) / 2.0
        loss_dict['loss_bg_pred'] = (self.regression_loss(pred, torch.zeros_like(pred), weight=bg_masks) + self.regression_loss(pred_os8, torch.zeros_like(pred_os8), weight=bg_masks)) / 2.0

        # FG losses between pred and pred_warp: rec loss, dtSSD loss
        loss_dict['loss_fg_rec'] = (self.regression_loss(pred_warp.detach(), pred, weight=fg_masks) + self.regression_loss(pred_warp_os8, pred_os8, weight=fg_masks)) / 2.0
        loss_dict['loss_fg_dtSSD'] = (loss_dtSSD(pred_warp.detach(), pred, mask=fg_masks) + loss_dtSSD(pred_warp_os8, pred_os8, mask=fg_masks)) / 2.0
        
        # Detail loss
        detail_mask = (detail_pred_warp > 1.0/255.0) & (detail_pred_warp < 254.0/255.0)
        detail_mask = self.dilate_mask(detail_mask.float())
        detail_mask = fg_masks * detail_mask
        
        # Rec loss
        loss_dict['loss_detail_pred_rec'] = self.regression_loss(pred_warp, detail_pred_warp, weight=detail_mask)
        loss_dict['loss_detail_warped_rec'] = self.regression_loss(pred, detail_pred_warp, weight=detail_mask)

        # Lap loss
        h, w = pred_warp.shape[-2:]
        loss_dict['loss_detail_pred_lap'] = self.lap_loss(pred_warp.view(-1, 1, h, w), detail_pred_warp.view(-1, 1, h, w), weight=detail_mask.view(-1, 1, h, w))
        loss_dict['loss_detail_warped_lap'] = self.lap_loss(pred.view(-1, 1, h, w), detail_pred_warp.view(-1, 1, h, w), weight=detail_mask.view(-1, 1, h, w))

        # Grad loss
        loss_dict['loss_detail_pred_grad'] = self.grad_loss(pred_warp.view(-1, 1, h, w), detail_pred_warp.view(-1, 1, h, w), mask=detail_mask.view(-1, 1, h, w))
        loss_dict['loss_detail_warped_grad'] = self.grad_loss(pred.view(-1, 1, h, w), detail_pred_warp.view(-1, 1, h, w), mask=detail_mask.view(-1, 1, h, w))
        
        loss_dict['total'] = loss_dict['loss_bg_warp'] \
                + loss_dict['loss_bg_pred'] \
                + loss_dict['loss_fg_rec'] * 0.5 \
                + loss_dict['loss_fg_dtSSD'] * 0.5 \
                + loss_dict['loss_detail_pred_rec'] * 0.25 \
                + loss_dict['loss_detail_warped_rec'] * 0.25 \
                + loss_dict['loss_detail_pred_lap'] * 0.2 \
                + loss_dict['loss_detail_warped_lap'] * 0.2 \
                + loss_dict['loss_detail_pred_grad'] \
                + loss_dict['loss_detail_warped_grad']

        return real_out, loss_dict

        loss_dict = {}
        loss_dict['loss_mo_os1'] = self.custom_regression_loss(alpha_pred_warp['alpha_os1'].detach(), alpha_pred['alpha_os1'], weight=weight) # F.l1_loss(alpha_pred_warp['alpha_os1'], alpha_pred['alpha_os1'], reduction='sum') / weight.sum()
        # weight = alpha_pred_warp['alpha_os4'].sum((-1, -2), keepdim=True)
        # loss_dict['loss_mo_os4'] = F.l1_loss(alpha_pred_warp['alpha_os4'], alpha_pred['alpha_os4'], reduction='sum') / weight.sum()
        loss_dict['loss_mo_os4'] = self.custom_regression_loss(alpha_pred_warp['alpha_os4'].detach(), alpha_pred['alpha_os4'], weight=weight)
        # weight = alpha_pred_warp['alpha_os8'].sum((-1, -2), keepdim=True)
        # loss_dict['loss_mo_os8'] = F.l1_loss(alpha_pred_warp['alpha_os8'], alpha_pred['alpha_os8'], reduction='sum') / weight.sum()
        loss_dict['loss_mo_os8'] = self.custom_regression_loss(alpha_pred_warp['alpha_os8'].detach(), alpha_pred['alpha_os8'], weight=weight)
        loss_dict['loss_mo'] = (loss_dict['loss_mo_os1'] * 2 + loss_dict['loss_mo_os4'] + loss_dict['loss_mo_os8']) / 5.0
        loss_dict['total'] = loss_dict['loss_mo']

        return real_out, loss_dict