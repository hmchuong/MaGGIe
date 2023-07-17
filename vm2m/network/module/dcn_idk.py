import torch
import torch.nn as nn
from .dcnv2 import DeformableConv2d
from vm2m.utils import resizeAnyShape

class DCInstDynKernelGenerator(nn.Module):
    def __init__(self, in_channel, in_dcn, n_inc_params, n_pix_params):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_dcn, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_dcn),
            nn.ReLU()
        )
        self.dcn = DeformableConv2d(in_dcn, in_dcn)
        self.smooth = nn.Sequential(
            nn.Linear(in_dcn, in_dcn),
            nn.ReLU()
        )   
        self.inc_layer = nn.Linear(in_dcn, n_inc_params, bias=False)
        self.pix_layer = nn.Linear(in_dcn, n_pix_params, bias=False)

    def forward(self, x, masks):
        '''
        x: (b*n_f) x c x h x w
        masks: b x n_f x n_i x H x W
        '''

        # FFW the features to DCN
        x = self.conv1(x)
        x = self.dcn(x)

        # Pooling feat to (b x n_i x in_dcn)
        b, n_f, n_i, H, W = masks.shape
        h, w = x.shape[2:]
        
        masks = resizeAnyShape(masks, size=(h, w))
        x = x.view(b, n_f, 1, -1, h * w)
        masks = masks.reshape(b, n_f, n_i, 1, h * w)
        x = (x * masks).sum(dim=-1) / (masks.sum(dim=-1) + 1e-6) # (b x n_f x n_i x in_dcn)
        
        x = self.smooth(x) # (b x n_f x n_i x in_dcn)

        inc_kernels = self.inc_layer(x) # (b x n_f x n_i x n_inc_params)
        pix_kernels = self.pix_layer(x) # (b x n_f x n_i x n_pix_params)

        return inc_kernels, pix_kernels


        