from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mgm import MGM
from vm2m.network.loss import loss_dtSSD

class TCVOM(MGM):
    def __init__(self, backbone, decoder, cfg):
        super(TCVOM, self).__init__(backbone, decoder, cfg)
        self.eval()
        self.dilate_op = nn.MaxPool2d(15, stride=1, padding=15//2)
        self.loss_dtSSD_w = cfg.loss_dtSSD_w
        self.loss_atten_w = cfg.loss_atten_w

        for param in self.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = True
        for param in self.decoder.layer1.parameters():
            param.requires_grad = False
        for param in self.decoder.layer2.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.encoder.train(False)
            self.aspp.train(False)

    def dilate(self, alpha, stride=15):
        mask = torch.logical_and(alpha>0.01, alpha<0.99).float()
        mask = self.dilate_op(mask)
        return mask
    
    def forward(self, batch):
        x = batch['image']
        masks = batch['mask']
        alphas = batch.get('alpha', None)
        trans_gt = batch.get('transition', None)
        fg = batch.get('fg', None)
        bg = batch.get('bg', None)

        # Combine input image and masks
        b, n_f, _, h, w = x.shape
        n_i = masks.shape[2]

        x = x.view(-1, 3, h, w)
        if masks.shape[-1] != w:
            masks = masks.view(-1, 1, h//8, w//8)
            masks = F.interpolate(masks, size=(h, w), mode="nearest")
        else:
            masks = masks.view(-1, 1, h, w)
            
        inp = torch.cat([x, masks], dim=1)
        if fg is not None:
            fg = fg.view(-1, 3, h, w)
        if bg is not None:
            bg = bg.view(-1, 3, h, w)
       
        embedding, mid_fea = self.encoder(inp)
        embedding = self.aspp(embedding)

        mid_fea = mid_fea['shortcut']

        # 1st forward: extract features and get first alphas
        raw_preds, features = self.decoder(embedding, mid_fea)[:2]
        
        preds = defaultdict(list)
        attb = [None] * n_f
        attf = [None] * n_f
        small_mask = [None] * n_f

        # Reshape features
        features = features.view(b, n_f, -1, *features.shape[-2:])
        embedding = embedding.view(b, n_f, -1, *embedding.shape[-2:])
        mid_fea = [f.view(b, n_f, -1, *f.shape[-2:]) for f in mid_fea]
        
        # Get unknown masks
        unknown_masks = self.dilate(raw_preds['alpha_os1'])
        unknown_masks = unknown_masks.view(b, n_f, -1, *unknown_masks.shape[-2:])

        # Add first frame to the list
        for k, v in raw_preds.items():
            preds[k].append(v.view(b, n_f, -1, *v.shape[-2:])[:, 0])

        
        # Forward middle frames
        for i in range(1, n_f - 1):
            curr_mid_feat =[f[:, i] for f in mid_fea]
            pred, _, attb[i], attf[i], small_mask[i] = self.decoder(embedding[:, i], curr_mid_feat, xb=features[:, i-1], xf=features[:, i+1], mask=unknown_masks[:, i])
            for k, v in pred.items():
                preds[k].append(v)

        

        # Add last frame to list
        for k, v in raw_preds.items():
            preds[k].append(v.view(b, n_f, -1, *v.shape[-2:])[:, -1])

            # Concatenate all frames
            preds[k] = torch.cat(preds[k], dim=1).view(-1, 1, h, w)

        

        # Fuse all predictions
        alpha_pred, weight_os4, weight_os1 = self.fushion(preds)

        output = {}
        output['alpha_os1'] = preds['alpha_os1'].view(b, n_f, n_i, h, w)
        output['alpha_os4'] = preds['alpha_os4'].view(b, n_f, n_i, h, w)
        output['alpha_os8'] = preds['alpha_os8'].view(b, n_f, n_i, h, w)

        alpha_pred = alpha_pred.view(b, n_f, n_i, h, w)
        output['refined_masks'] = alpha_pred

        if self.training:
            reshaped_alphas = alphas.view(-1, 1, h, w)
            reshaped_trans_gt = trans_gt.view(-1, 1, h, w)
            iter = batch['iter']

            # Compute image loss
            loss_dict = self.compute_loss(preds, weight_os4, weight_os1, reshaped_alphas, reshaped_trans_gt, fg, bg, iter, alphas.shape)

            # Compute attention loss
            if self.loss_atten_w > 0:
                attn_loss = self.compute_atten_loss(alphas, attb, attf, small_mask)
                loss_dict['loss_atten'] = attn_loss
                total_loss = loss_dict['total'] + attn_loss * self.loss_atten_w
                loss_dict['total'] = total_loss


            return output, loss_dict
        return output

    def compute_atten_loss(self, alphas, attb, attf, small_mask):
        
        os = 8
        bs, n_f, _, h, w = alphas.shape
        l_att = []
        h = h // os
        w = w // os
        for c in range(1, n_f - 1):
            bgt = F.avg_pool2d(alphas[:, c-1], os, os)
            fgt = F.avg_pool2d(alphas[:, c+1], os, os)
            cgt = F.avg_pool2d(alphas[:, c], os, os)
            m = small_mask[c].reshape(bs, -1)
            if m.float().sum() == 0:
                l_att.append(torch.tensor(0.0).to(attb[c].device))
                continue
            b = attb[c].reshape(bs, -1, h * w).permute(1, 0, 2)
            f = attf[c].reshape(bs, -1, h * w).permute(1, 0, 2)
            cb = b[:, m]
            cf = f[:, m]

            # Construct groundtruths
            with torch.no_grad():
                bgt_unfold = F.unfold(bgt, kernel_size=9, padding=9//2).reshape(bs, -1, h * w).permute(1, 0, 2)
                fgt_unfold = F.unfold(fgt, kernel_size=9, padding=9//2).reshape(bs, -1, h * w).permute(1, 0, 2)
                cgt = cgt.reshape(bs, 1, h * w).permute(1, 0, 2)
                bgt_unfold = bgt_unfold[:, m]
                fgt_unfold = fgt_unfold[:, m]
                cgt = cgt[:, m]

                dcb = torch.abs(cgt - bgt_unfold)
                dcb = (dcb < 0.3).float() * (1 - 0.2)
                dcf = torch.abs(cgt - fgt_unfold)
                dcf = (dcf < 0.3).float() * (1 - 0.2)
            loss = F.binary_cross_entropy_with_logits(cb, dcb) + F.binary_cross_entropy_with_logits(cf, dcf)
            l_att.append(loss / 2.0)
        l_att = sum(l_att) / float(len(l_att))
        return l_att

