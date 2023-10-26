import cv2
import torch
from torch.nn import functional as F
import albumentations as A
from .mgm_tempspar import MGM_TempSpar

class MGM_SS(MGM_TempSpar):
    def __init__(self, backbone, decoder, cfg):
        super().__init__(backbone, decoder, cfg)
        
        # Add the motion consistency loss
        self.motion_aug = A.MotionBlur(p=1.0, blur_limit=(3, 49))

        kernel_size = 30
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
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
        batch["weight"] = torch.cat([batch["weight"], torch.zeros_like(batch["weight"])], dim=0)
        batch["transition"] = torch.cat([batch["transition"], torch.zeros_like(batch["transition"])], dim=0)

        # batch = self.flatten_batch(batch)
        # batch["image"] = torch.cat([batch["image"], new_images.flatten(0,1)[:, None]], dim=0)
        # batch["mask"] = torch.cat([batch["mask"], new_masks.flatten(0,1)[:, None]], dim=0)
        # batch["alpha"] = torch.cat([batch["alpha"], torch.zeros_like(batch["alpha"])], dim=0)
        # batch["transition"] = torch.cat([batch["transition"], torch.zeros_like(batch["transition"])], dim=0)
        # for k in ['weight', 'fg', 'bg']:
        #     if k in batch:
        #         del batch[k]
        # return batch

    def warp_alpha(self, out, kernel):
        # Combine OS8, OS4, and OS1: (b, 1, n_i, h, w)
        alpha_os1 = out['alpha_os1']
        alpha_os4 = out['alpha_os4']
        alpha_os8 = out['alpha_os8']
        refined_masks = out['refined_masks']
        n_samples = alpha_os1.shape[0]

        alpha_pred = {
            "alpha_os1": alpha_os1[n_samples//2:],
            "alpha_os4": alpha_os4[n_samples//2:],
            "alpha_os8": alpha_os8[n_samples//2:],
            "refined_masks": refined_masks[n_samples//2:],
        }

        # warp alpha
        b, n_f, n_i, h, w = alpha_os1.shape
        inputs = torch.cat([alpha_os1[:n_samples//2], alpha_os4[:n_samples//2], alpha_os8[:n_samples//2], refined_masks[:n_samples//2]], dim=1)
        inputs = inputs.flatten(0,1)
        kernel = kernel.type_as(inputs)
        kernel = kernel.expand(inputs.shape[1], 1, -1, -1)
        
        # Apply motion blur
        out = F.conv2d(inputs, kernel, padding=kernel.shape[-1]//2, groups=inputs.shape[1])
        out = out.view(n_samples//2, -1, n_i, h, w)
        alpha_pred_warp = {
            "alpha_os1": out[:, :n_f],
            "alpha_os4": out[:, n_f:2*n_f],
            "alpha_os8": out[:, 2*n_f:3*n_f],
            "refined_masks": out[:, 3*n_f:]
        }
        return alpha_pred_warp, alpha_pred
    
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

    def forward(self, batch, is_real=False,**kwargs):
        
        # if not is_real:
        #     return super().forward(batch, **kwargs)
        
        
        # iter = batch["iter"]

        # # Divide input into two parts: with gt and without gt
        # syn_data_mask = batch["is_syn"]
        # real_data_mask = ~syn_data_mask
        
        
        # if syn_data_mask.sum() > 0:
        #     # For the input with gt, compute supervised loss for the input with gt
        #     syn_batch = self.extract_batch(batch, syn_data_mask)
        #     syn_batch["iter"] = iter
        #     syn_out, loss_dict = super().forward(syn_batch, return_ctx=False, mem_feat=[], mem_query=None, mem_details=None, **kwargs)
        # else:
        #     syn_out = {}
        #     loss_dict = {}
        #     loss_dict['total'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_rec_os1'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_rec_os4'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_rec_os8'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_rec'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_lap_os1'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_lap_os4'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_lap_os8'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_lap'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_multi_inst'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_dtSSD_os1'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_dtSSD_os4'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_dtSSD_os8'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_dtSSD'] = torch.zeros(1).type_as(batch["image"])
        #     loss_dict['loss_max_atten'] = torch.zeros(1).type_as(batch["image"])

        # if real_data_mask.sum() == 0:
        #     loss_dict['loss_mo_os1'] = torch.zeros(1).type_as(syn_out['alpha_os1'])
        #     loss_dict['loss_mo_os4'] = torch.zeros(1).type_as(syn_out['alpha_os4'])
        #     loss_dict['loss_mo_os8'] = torch.zeros(1).type_as(syn_out['alpha_os8'])
        #     loss_dict['loss_mo'] = torch.zeros(1).type_as(syn_out['alpha_os1'])
        #     return syn_out, loss_dict
        # # Apply motion blur to the input without gt and generate additional input
        # motion_kernel = self.motion_aug.get_params()["kernel"]
        # motion_kernel = torch.from_numpy(motion_kernel).to(batch["image"].device)

        # motioned_images, motioned_masks = self.generate_motion_input(batch["image"][real_data_mask], batch["mask"][real_data_mask], motion_kernel)

        # # For the input without gt, compute supervised loss for the input without gt
        # real_batch = self.extract_batch(batch, real_data_mask)
        # real_batch = self.prepare_real_batch(real_batch, motioned_images, motioned_masks)
        
        # real_batch["iter"] = iter
        # real_out, _ = super().forward(real_batch, return_ctx=False, mem_feat=[], mem_query=None, mem_details=None, **kwargs)
        
        # # Compute motioned alpha with motion kernel
        # alpha_pred_warp, alpha_pred = self.warp_alpha(real_out, motion_kernel)

        # # Compute motion consistency loss for the input without gt
        # weight = alpha_pred_warp['alpha_os1'].sum((-1, -2), keepdim=True)
        # loss_dict['loss_mo_os1'] = F.l1_loss(alpha_pred_warp['alpha_os1'], alpha_pred['alpha_os1'], reduction='sum') / weight.sum()
        # weight = alpha_pred_warp['alpha_os4'].sum((-1, -2), keepdim=True)
        # loss_dict['loss_mo_os4'] = F.l1_loss(alpha_pred_warp['alpha_os4'], alpha_pred['alpha_os4'], reduction='sum') / weight.sum()
        # weight = alpha_pred_warp['alpha_os8'].sum((-1, -2), keepdim=True)
        # loss_dict['loss_mo_os8'] = F.l1_loss(alpha_pred_warp['alpha_os8'], alpha_pred['alpha_os8'], reduction='sum') / weight.sum()
        # loss_dict['loss_mo'] = (loss_dict['loss_mo_os1'] * 2 + loss_dict['loss_mo_os4'] + loss_dict['loss_mo_os8']) / 5.0
        # loss_dict['total'] = loss_dict['total'] + loss_dict['loss_mo'] * 0.25

        # # Merge the output
        # if len(syn_out) > 0:
        #     merged_out = self.merge_output(syn_out, real_out, real_data_mask)
        # else:
        #     merged_out = real_out
        # return merged_out, loss_dict
    
        if not is_real:
            return super().forward(batch, **kwargs)
        
        # import pdb; pdb.set_trace()
        
        iter = batch["iter"]
        valid_masks = batch['mask'].sum((-1, -2), keepdim=True) > 0
        
        # Apply motion blur to the input without gt and generate additional input
        motion_kernel = self.motion_aug.get_params()["kernel"]
        # import pickle
        # device_id = batch["image"].device.index
        # pickle.dump(motion_kernel, open(f"motion_kernel_{device_id}.pkl", "wb"))

        motion_kernel = torch.from_numpy(motion_kernel).to(batch["image"].device)

        motioned_images, motioned_masks = self.generate_motion_input(batch["image"], batch["mask"], motion_kernel)

        # For the input without gt, compute supervised loss for the input without gt
        # real_batch = self.extract_batch(batch, real_data_mask)
        real_batch = batch

        # Add motioned images and masks to the batch
        self.prepare_real_batch(real_batch, motioned_images, motioned_masks)
        
        real_batch["iter"] = iter
        
        real_out, _ = super().forward(real_batch, **kwargs)
        
        # Compute motioned alpha with motion kernel
        alpha_pred_warp, alpha_pred = self.warp_alpha(real_out, motion_kernel)

        # Compute motion consistency loss for the input without gt
        # weight = alpha_pred_warp['alpha_os1'].sum((-1, -2), keepdim=True)
        # import pdb; pdb.set_trace()

        # weight = torch.ones_like(alpha_pred_warp['alpha_os1'])
        # weight = weight * valid_masks

        # Combine OS8, OS4, and OS1
        pred_warp = alpha_pred_warp["refined_masks"]
        pred = alpha_pred["refined_masks"]
        
        # FG/ Unknown mask: Dilate the input masks and union them
        dilated_masks = self.dilate_mask(real_batch['mask'])
        dilated_masks = dilated_masks.sum(0, keepdim=True)
        # cv2.imwrite("test_dilate_mask.png", real_batch['mask'][0,0, 2].cpu().numpy()* 255) 
        # cv2.imwrite("test_dilate_mask.png", dilated_masks[0,0, 2].cpu().numpy()* 255)
        # import pdb; pdb.set_trace()

        # BG loss outside the mask region to remove noise
        bg_masks = (dilated_masks < 0.5).float()

        # FG loss inside the mask region
        loss_dict = {}
        loss_dict['loss_fg'] = self.custom_regression_loss(pred_warp, pred, weight=dilated_masks)
        loss_dict['loss_bg_warp'] = self.custom_regression_loss(pred_warp, torch.zeros_like(pred_warp), weight=bg_masks)
        loss_dict['loss_bg_pred'] = self.custom_regression_loss(pred, torch.zeros_like(pred), weight=bg_masks)
        
        loss_dict['total'] = loss_dict['loss_fg'] + loss_dict['loss_bg_warp'] + loss_dict['loss_bg_pred']

        return real_out, loss_dict

        # import cv2
        # Check images
        # for i_f in range(weight.shape[1]):
        #     image_before = real_batch['image'][0,i_f]
        #     image_after = real_batch['image'][1,i_f]
        #     denorm_image_before = image_before * torch.tensor([0.229, 0.224, 0.225], device=image_before.device).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406], device=image_before.device).view(3,1,1)
        #     denorm_image_after = image_after * torch.tensor([0.229, 0.224, 0.225], device=image_after.device).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406], device=image_after.device).view(3,1,1)
        #     denorm_image_before = denorm_image_before.permute(1,2,0).cpu().numpy()
        #     denorm_image_after = denorm_image_after.permute(1,2,0).cpu().numpy()
        #     import cv2
        #     cv2.imwrite(f"image_before.png", denorm_image_before[:, :, ::-1] * 255)
        #     cv2.imwrite(f"image_after.png", denorm_image_after[:, :, ::-1] * 255)
        #     import pdb; pdb.set_trace()

        # Check masks

        # for i_f in range(weight.shape[1]):
        #     valid_mask_ids = torch.nonzero(weight.sum((0, 1, 3, 4))).flatten()
        #     for id in valid_mask_ids:
        #         mask_before = real_batch['mask'][0,i_f,id]
        #         mask_after = real_batch['mask'][1,i_f,id]
        #         alpha_before = real_out['alpha_os8'][0,i_f,id]
        #         alpha_after = real_out['alpha_os8'][1,i_f,id]
        #         alpha_correct =  alpha_pred_warp['alpha_os8'][0,i_f,id]
        #         alpha_pred_motion = alpha_pred['alpha_os8'][0,i_f,id]
        #         denorm_mask_before = mask_before.float().cpu().numpy()
        #         denorm_mask_after = mask_after.float().cpu().numpy()
        #         denorm_alpha_before = alpha_before.float().detach().cpu().numpy()
        #         denorm_alpha_after = alpha_after.float().detach().cpu().numpy()
        #         denorm_alpha_correct = alpha_correct.float().detach().cpu().numpy()
        #         denorm_alpha_pred_motion = alpha_pred_motion.float().detach().cpu().numpy()
                
        #         cv2.imwrite(f"mask_before.png", denorm_mask_before * 255)
        #         cv2.imwrite(f"mask_after.png", denorm_mask_after * 255)
        #         cv2.imwrite(f"alpha_before.png", denorm_alpha_before * 255)
        #         cv2.imwrite(f"alpha_after.png", denorm_alpha_after * 255)
        #         cv2.imwrite(f"alpha_correct.png", denorm_alpha_correct * 255)
        #         cv2.imwrite(f"alpha_pred_motion.png", denorm_alpha_pred_motion * 255)
        #         import pdb; pdb.set_trace()

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