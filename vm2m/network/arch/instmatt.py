import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .mgm import MGM_SingInst, MGM

class InstMatt(MGM):
    def __init__(self, backbone, decoder, cfg):
        super().__init__(backbone, decoder, cfg)

        # Init refinement module
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        inc = 32
        ouc = 16
        self.split = nn.Conv2d(inc*3, ouc*3, kernel_size=3, padding=1, groups=3)
        self.merge = nn.Sequential(
            nn.Conv2d(ouc*3+3, inc, kernel_size=3, padding=1),
            self.leaky_relu,
            nn.Conv2d(inc, inc, kernel_size=3, padding=1),
        )
        self.head = nn.Conv2d(inc, 3, kernel_size=3, padding=1)
        self.ouc = ouc
        self.is_training = True
        self.train()
    
    def train(self, mode: bool = True):
        super().train(False)
        self.is_training = mode
        # Train only the InstMatt module
        # import pdb; pdb.set_trace()
        if mode:
            for param in self.parameters():
                param.requires_grad = False
            # n_param = sum([p.numel() for p in self.parameters() if p.requires_grad])
            for module in [self.leaky_relu, self.split, self.merge, self.head]:
                module.train(True)
                for p in module.parameters():
                    p.requires_grad = True
            # n_param_new = sum([p.numel() for p in self.parameters() if p.requires_grad])
            # import pdb; pdb.set_trace()
        return self

    def get_boxes(self, distance, win):
        oh, ow = distance.shape[2:]
        distance = F.interpolate(distance, scale_factor=0.1, mode="nearest").squeeze() # hxw
        h, w = distance.shape
        xs = np.arange(w)
        ys = np.arange(h)
        xaxis, yaxis = np.meshgrid(xs, ys)
        xy = np.stack([xaxis, yaxis], axis=2)
        xy = np.reshape(xy, (-1, 2))
        boxes = np.tile(xy, (1, 2)) * 10
        scores = distance.reshape((-1,1)).data.cpu().numpy()
        valid_boxes = boxes + np.array([-win//2, -win//2, win//2, win//2])
        off_x1 =  - np.minimum(valid_boxes[:,0],0) + np.minimum(ow-valid_boxes[:,0],0)
        off_y1 =  - np.minimum(valid_boxes[:,1],0) + np.minimum(oh-valid_boxes[:,1],0)
        off_x2 =  - np.minimum(valid_boxes[:,2],0) + np.minimum(ow-valid_boxes[:,2],0)
        off_y2 =  - np.minimum(valid_boxes[:,3],0) + np.minimum(oh-valid_boxes[:,3],0)
        valid_boxes[:,0] = valid_boxes[:,0] + off_x1 + off_x2
        valid_boxes[:,1] = valid_boxes[:,1] + off_y1 + off_y2
        valid_boxes[:,2] = valid_boxes[:,2] + off_x1 + off_x2
        valid_boxes[:,3] = valid_boxes[:,3] + off_y1 + off_y2
        dets = np.concatenate([valid_boxes, scores], axis=1)
        valid_dets = dets[scores[:,0]>0.01]
        if len(valid_dets) == 0:
            keep_idx =  dets[:,-1].argsort()[::-1][0:1]
            keep_boxes = dets[keep_idx]
        else:
            keep_idx = nms(valid_dets, 0.3, win)
            keep_boxes = valid_dets[keep_idx]
        return keep_boxes

    def is_same_instance(self, pred):
        a = pred.clone() # kx1xhxw
        b = pred.squeeze(1).unsqueeze(0) # 1xkxhxw
        if pred.size(0) > 8:
            a = F.interpolate(a, scale_factor=0.5)
            b = F.interpolate(b, scale_factor=0.5)
        intersection = torch.logical_and(a>0.01, b>0.01).float()
        union = torch.logical_or(a>0.01, b>0.01).float()
        minregion = torch.min((a>0.01).float().sum(dim=(2,3)), (b>0.01).float().sum(dim=(2,3)))
        suma = (a>0.01).float().sum(dim=(2,3))
        mad = ((a-b).abs() * intersection).sum(dim=(2,3)) / (intersection.sum(dim=(2,3)) + 1.)
        iou = intersection.sum(dim=(2,3)) / (minregion + 1.)
        ioua = intersection.sum(dim=(2,3)) / (suma + 1.)
        for i in range(iou.size(0)):
            iou[i,i] = 0
            ioua[i,i] = 0
        bad = ((iou > 0.3).sum(dim=1)>1).float()
        for i in range(iou.size(0)):
            for j in range(i+1, iou.size(1)):
                if iou[i,j]>0.5 or iou[j,i]>0.5:
                    if iou[i,j] > iou[j,i]:
                        bad[i] = 1
                    else:
                        bad[j] = 1
        return bad
    
    def prepare_trimatte(self, mask):
        mask_batch = []
        for i in range(mask.shape[0]):
            mask_t = mask[i]
            mask_r = torch.sum(mask, dim=0) - mask_t
            mask_b = 1 - torch.sum(mask, dim=0)
            new_mask = torch.stack([mask_t, mask_r, mask_b], dim=0)
            mask_batch.append(new_mask)
        mask_batch = torch.stack(mask_batch, dim=0)
        return mask_batch

    def forward(self, batch, **kwargs):
        # feat_list: K, C, H, W
        # pred_list: K, 3, H, W (target mask, ref mask, bg mask)
        nostop = True

        
        # Transform batch to K, 3, H, W
        image = batch['image'] # 1 x 1 x 3 x H x W
        mask = batch['mask'] # 1 x 1 x N x H x W
        alphas = batch.get('alpha', None)
        valid_mask = mask.sum((3, 4)) > 0
        mask = mask[:, :, valid_mask[0, 0]] # 1 x 1 x K x H x W
        mask = mask[0,0] # K x H x W
        if alphas is not None:
            alphas = alphas[:, :, valid_mask[0, 0]] # 1 x 1 x K x H x W
            alphas = alphas[0,0] # K x H x W
        
        mask_batch = self.prepare_trimatte(mask)
        if alphas is not None:
            alpha_batch = self.prepare_trimatte(alphas)
        # import pdb; pdb.set_trace()
        # alpha_batch = self.prepare_trimatte

        image = image.repeat(mask_batch.size(0), 1, 1, 1, 1)
        mask_batch = mask_batch.reshape(-1, 1, 3, mask.size(-2), mask.size(-1)) # K x 1 x 3 x H x W

        # Predict the matte
        with torch.no_grad():
            pred = super().forward({'image': image, 'mask': mask_batch}, **kwargs)
        # import pdb; pdb.set_trace()
        # Refine the matte
        # Convert pred_list to K x 3 x H x W
        pred_list = pred['refined_masks'].clone().detach()
        feat_list = pred['feature'].clone().detach()
        pred_list = pred_list.flatten(0,1)
        x = image.reshape(-1, 3, image.size(-2), image.size(-1))
        # weight_os1 = pred['weight_os1'].reshape(-1, 3, image.size(-2), image.size(-1))
        

        if not self.is_training:
            
            # same instance?
            bad = self.is_same_instance(pred_list[:, 0:1, :, :])
            bad = bad.long()

        if self.is_training:
            win = 128
        else:
            if pred_list.size(0) <=8:
                win = 320
            elif pred_list.size(0) <= 14:
                win = 256
            else:
                win = 128
        
        k, _, h, w = pred_list.shape
        _, c, _, _ = feat_list.shape

        win = min(win, min(h,w))

        # update B
        B_cat = pred_list[:,2:3].mean(dim=0, keepdim=True) # 1x1xhxw
        T_cat = pred_list[:,0:1].sum(dim=0, keepdim=True) # 1x1xhxw
        distance = (1-T_cat-B_cat).abs() # hxw
        boxes = self.get_boxes(distance, win)

        if self.is_training:
            boxes = boxes[0:1]

        if self.is_training:
            iteration = random.randint(1,3)
        else:
            iteration = 3

        if nostop:
            refine_list = []

            for _ in range(iteration):
                for box in boxes:
                    x1, y1, x2, y2, score = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if self.is_training:
                        pred_patch = pred_list[:,:,y1:y2,x1:x2]
                        feat_patch = feat_list[:,:,y1:y2,x1:x2]
                        x_patch = x[:,:,y1:y2,x1:x2]
                    else:
                        pred_patch = pred_list[(1-bad)>0,:,y1:y2,x1:x2]
                        feat_patch = feat_list[(1-bad)>0,:,y1:y2,x1:x2]
                        x_patch = x[(1-bad)>0,:,y1:y2,x1:x2]
                    _, _, ph, pw = pred_patch.shape

                    # split
                    attention = torch.cat([pred_patch[:,0:1].repeat(1,c,1,1),
                                           pred_patch[:,1:2].repeat(1,c,1,1),
                                           pred_patch[:,2:3].repeat(1,c,1,1)], dim=1)
                    feat_split = self.split(feat_patch.repeat(1,3,1,1) * attention) # cx3

                    # change information
                    feat_t = feat_split[:,self.ouc*1:self.ouc*2].sum(dim=0, keepdim=True) / max((feat_split.size(0)-1),1) - \
                             feat_split[:,self.ouc*0:self.ouc*1].sum(dim=0, keepdim=True) + \
                             feat_split[:,self.ouc*0:self.ouc*1] * 2
                    feat_r = feat_split[:,self.ouc*0:self.ouc*1].sum(dim=0, keepdim=True) - feat_split[:,self.ouc*0:self.ouc*1] + \
                             feat_split[:,self.ouc*1:self.ouc*2]
                    feat_b = feat_split[:,self.ouc*2:self.ouc*3].mean(dim=0, keepdim=True).repeat(feat_split.size(0),1,1,1)
                    feat_split = torch.cat([feat_t, feat_r, feat_b, x_patch], dim=1)

                    # merge
                    feat_merge = self.merge(feat_split)
                    pred = self.head(feat_merge)
                    if self.is_training:
                        pred_list[:,:,y1+10:y2-10,x1+10:x2-10] = (torch.tanh(pred[:,:,10:ph-10,10:pw-10])+1.0)/2.0
                    else:
                        pred_list[(1-bad)>0,:,y1+10:y2-10,x1+10:x2-10] = (torch.tanh(pred[:,:,10:ph-10,10:pw-10])+1.0)/2.0
                refine_list.append(pred_list)
        else:
            refine_list = [pred_list.clone()]

            state = torch.ones(bad.size()).long().to(bad.device)
            last_pred = pred_list.clone()
            while True:
                for box in boxes:
                    x1, y1, x2, y2, score = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    pred_patch = pred_list[(1-bad)>0,:,y1:y2,x1:x2]
                    feat_patch = feat_list[(1-bad)>0,:,y1:y2,x1:x2]
                    x_patch = x[(1-bad)>0,:,y1:y2,x1:x2]
                    _, _, ph, pw = pred_patch.shape

                    # split
                    attention = torch.cat([pred_patch[:,0:1].repeat(1,c,1,1),
                                           pred_patch[:,1:2].repeat(1,c,1,1),
                                           pred_patch[:,2:3].repeat(1,c,1,1)], dim=1)
                    feat_split = self.split(feat_patch.repeat(1,3,1,1) * attention) # cx3

                    # change information
                    feat_t = feat_split[:,self.ouc*1:self.ouc*2].sum(dim=0, keepdim=True) / max((feat_split.size(0)-1),1) - \
                             feat_split[:,self.ouc*0:self.ouc*1].sum(dim=0, keepdim=True) + \
                             feat_split[:,self.ouc*0:self.ouc*1] * 2
                    feat_r = feat_split[:,self.ouc*0:self.ouc*1].sum(dim=0, keepdim=True) - feat_split[:,self.ouc*0:self.ouc*1] + \
                             feat_split[:,self.ouc*1:self.ouc*2]
                    feat_b = feat_split[:,self.ouc*2:self.ouc*3].mean(dim=0, keepdim=True).repeat(feat_split.size(0),1,1,1)
                    feat_split = torch.cat([feat_t, feat_r, feat_b, x_patch], dim=1)

                    # merge
                    feat_merge = self.merge(feat_split)
                    pred = (torch.tanh(self.head(feat_merge)[:,:,10:ph-10,10:pw-10]) + 1.0) / 2.0
                    pred_list[((1-bad)*state)>0,:,y1+10:y2-10,x1+10:x2-10] = pred[((state*(1-bad))[(1-bad)>0])>0]
                diff = (pred_list[:,0:1] - last_pred[:,0:1]).abs().amax(dim=(1,2,3))
                state = state * (diff>0.01)
                refine_list.append(pred_list.clone())
                last_pred = pred_list
                if state.sum() == 0:
                    break
        
        output_dict = {}
        output = refine_list[-1][:, 0:1]
        
        if self.is_training:
            padded_output = torch.zeros(1, 1, 10, h, w, device=output.device, dtype=output.dtype)
            padded_output[:, :, valid_mask[0, 0]] = output.reshape(1, 1, -1, h, w)
            # import pdb; pdb.set_trace()
            output_dict['refined_masks'] = padded_output
        else:
            output_dict['refined_masks'] = output.reshape(1, 1, -1, h, w)

        # import pdb; pdb.set_trace()

        # Compute loss
        # refine_list[0]: K x 3 x H x W
        if self.is_training:
            # Compute loss
            loss_dict = {}
            total_loss = 0

            weight = torch.zeros_like(alpha_batch)
            for box in boxes:
                x1, y1, x2, y2, score = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                weight[:,:,y1:y2,x1:x2] = 1
            # Rec loss
            if self.loss_alpha_w > 0:
                ref_rec = 0
                for i in range(len(refine_list)):
                    
                    # import cv2
                    # cv2.imwrite("pred.png", refine_list[i][0,2].detach().cpu().numpy() * 255)
                    # cv2.imwrite("gt.png", alpha_batch[0,2].cpu().numpy() * 255)
                    # cv2.imwrite("weight.png", weight[0, 0].cpu().numpy() * 255)
                    # import pdb; pdb.set_trace()
                    ref_rec += self.regression_loss(refine_list[i], alpha_batch, loss_type=self.cfg.loss_alpha_type, weight=weight)
                ref_rec /= len(refine_list)
                loss_dict['loss_ref_rec'] = ref_rec
                total_loss += self.loss_alpha_w * ref_rec
            
            # Lap loss
            if self.loss_alpha_lap_w > 0:
                ref_lap = 0
                for i in range(len(refine_list)):
                    ref_lap += self.lap_loss(refine_list[i].view(-1, 1, h, w), alpha_batch.view(-1, 1, h, w), weight.view(-1, 1, h, w))
                ref_lap /= len(refine_list)
                loss_dict['loss_ref_lap'] = ref_lap
                total_loss += self.loss_alpha_lap_w * ref_lap
            
            ref_constraint = 0
            for i in range(len(refine_list)):
                alpha_sum = torch.sum(refine_list[i], dim=1, keepdim=True)   
                ref_constraint += self.regression_loss(alpha_sum, torch.ones_like(alpha_sum), loss_type=self.cfg.loss_alpha_type, weight=weight)
            ref_constraint /= len(refine_list)
            loss_dict['loss_ref_const'] = ref_constraint
            total_loss += ref_constraint
            loss_dict['total'] = total_loss
            return output_dict, loss_dict

        return output_dict

def nms(dets, thresh, step=512):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        cx = (x1[order] + x2[order]) / 2
        cy = (y1[order] + y2[order]) / 2
        dx = np.abs(cx[0] - cx[1:])
        dy = np.abs(cy[0] - cy[1:])
        dist_xy = np.stack([dx, dy], axis=1)
        valid = np.all(dist_xy<np.array([step//2, step//2]), 1)
        inds = np.where(1-np.logical_and(ovr>thresh, valid))[0]
        order = order[inds + 1]
    return keep