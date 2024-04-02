import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAggregationModule(nn.Module):
    def __init__(self, input_chn, reduction, window):
        super(FeatureAggregationModule, self).__init__()
        out_chn = input_chn // reduction
        self.key_conv = nn.Conv2d(input_chn, out_chn, kernel_size=3, padding=1)
        self.query_conv = nn.Conv2d(input_chn, out_chn, kernel_size=3, padding=1)
        self.value_conv = nn.Conv2d(input_chn, out_chn, kernel_size=3, padding=1)
        self.window = window

    def forward(self, x, b, f, mask):
        # mask should be [B, 1, 8H, 8W] in {0, 1}
        # we'll resize it to the size of x here
        B, C, H, W = x.shape
        mask = F.interpolate(mask, size=(H, W), mode='nearest').bool()

        def _attention(q, target):
            k = self.key_conv(target)   # [B, C, H, W]
            assert q.shape == k.shape and k.shape == v.shape

            m = mask.reshape(B, -1)
            feats, atts = [], []
            for i in range(B):
                k_unfold = k[i].unsqueeze(0)
                if self.window == 1:
                    k_unfold = k_unfold.reshape(C, 1, H*W)
                else:
                    k_unfold = F.unfold(k_unfold, self.window, \
                        padding=self.window // 2).reshape(C, -1, H*W)       # [C, W**2, N]
                q_reshape = q[i]
                q_reshape = q_reshape.reshape(C, 1, -1)                 # [C, 1, N]
                
                mi = torch.nonzero(m[i], as_tuple=False).squeeze()      # [U]
                # we only care about trimap's unknown region
                k_unfold = k_unfold[..., mi]            # [C, W**2, U]
                q_reshape = q_reshape[..., mi]          # [C, 1, U]

                # dot
                qdotk = torch.sum(q_reshape * k_unfold, dim=0, keepdim=True) / math.sqrt(C)     # [1, W**2, U]
                qdotkN = torch.zeros(self.window**2, H*W).to(torch.cuda.current_device())       # [w**2, N]
                qdotkN[:, mi] = qdotk[0]
                atts.append(qdotkN)
                att = torch.softmax(qdotk, dim=1)                                                       # [1, W**2, U], weight

                # attentioned target feature based on query
                atted = torch.sum(att * k_unfold, dim=1)                        # [C, U]
                feat = torch.zeros(C, H*W).to(torch.cuda.current_device())      # [C, N]
                feat[:, mi] = atted
                feats.append(feat.reshape(C, H, W))

            # back to batch
            tgt_feat = torch.stack(feats)
            atts = torch.stack(atts)
            return tgt_feat, atts
        
        q = self.query_conv(x)
        v = self.value_conv(x)
        xb, attb = _attention(q, b)
        xf, attf = _attention(q, f)
        #       feature     B,w**2,H,W  B,1,H,W
        return v + xb + xf, attb, attf, mask