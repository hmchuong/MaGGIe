import cv2
import numpy as np
import torch
from torch.nn import functional as F
from kornia.morphology import dilation

def resizeAnyShape(x, scale_factor=None, size=None, mode='bilinear', align_corners=False, use_max_pool=False):
    shape = x.shape
    dtype = x.dtype
    x = x.view(-1, shape[-3], *shape[-2:]).float()
    if use_max_pool:
        assert scale_factor is not None, "scale_factor must be specified when use_max_pool=True"
        assert scale_factor < 1.0, "scale_factor must be less than 1.0 when use_max_pool=True"
        stride = int(1 / scale_factor)
        x = F.max_pool2d(x, kernel_size=stride, stride=stride)
    else:
        x = F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    x = x.view(*shape[:-2], *x.shape[-2:]).to(dtype)
    return x

Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]
def compute_unknown(masks, k_size=30, is_train=False):
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    h, w = masks.shape[-2:]
    uncertain = (masks > 1.0/255.0) & (masks < 254.0/255.0)
    ori_shape = uncertain.shape

    # ----- Using kornia -----
    # uncertain = uncertain.view(-1, 1, h, w)
    # kernel = torch.from_numpy(kernel).to(masks.device).float()
    # uncertain = dilation(uncertain.float(), kernel, engine='convolution')
    # uncertain = uncertain.view(*ori_shape)
    # uncertain = (uncertain > 0.0).float()

    # ----- Using cv2 -----
    uncertain = uncertain.view(-1, h, w).detach().cpu().numpy().astype('uint8')

    for n in range(uncertain.shape[0]):
        if is_train:
            width = np.random.randint(1, k_size)
        else:
            width = k_size // 2
        uncertain[n] = cv2.dilate(uncertain[n], Kernels[width])
    
    uncertain = uncertain.reshape(ori_shape)
    uncertain = torch.from_numpy(uncertain).to(masks.device)

    return uncertain