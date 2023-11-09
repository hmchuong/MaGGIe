import cv2
import numpy as np
import torch
from torch.nn import functional as F
from kornia.morphology import dilation

def resizeAnyShape(x, scale_factor=None, size=None, mode='bilinear', align_corners=False, use_max_pool=False, use_avg_pool_binary=False):
    shape = x.shape
    dtype = x.dtype
    x = x.view(-1, shape[-3], *shape[-2:]).float()
    if use_max_pool:
        assert scale_factor is not None, "scale_factor must be specified when use_max_pool=True"
        assert scale_factor < 1.0, "scale_factor must be less than 1.0 when use_max_pool=True"
        stride = int(1 / scale_factor)
        x = F.max_pool2d(x, kernel_size=stride, stride=stride)
    elif use_avg_pool_binary:
        assert scale_factor is not None, "scale_factor must be specified when use_avg_pool_w_threshold=True"
        assert scale_factor < 1.0, "scale_factor must be less than 1.0 when use_avg_pool_w_threshold=True"
        stride = int(1 / scale_factor)
        x = F.avg_pool2d(x, kernel_size=stride, stride=stride)
        x = (x > 0.0).float()
    else:
        x = F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    x = x.view(*shape[:-2], *x.shape[-2:]).to(dtype)
    return x

Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]
def compute_unknown(masks, k_size=30, is_train=False, lower_thres=1.0/255.0):
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    h, w = masks.shape[-2:]
    uncertain = (masks > lower_thres) & (masks < 254.0/255.0)
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




# Create a Gaussian kernel
def gaussian_kernel(size, sigma):
    grid = torch.arange(size).float() - size // 2
    gaussian = torch.exp(-grid**2 / (2 * sigma**2))
    gaussian /= gaussian.sum()
    return gaussian.view(1, 1, -1) * gaussian.view(1, 1, -1)

def gaussian_smoothing(input_tensor, sigma):

    kernel_size = sigma * 2 + 1

    # Apply padding
    padding = kernel_size // 2
    padded_tensor = F.pad(input_tensor, (padding, padding, padding, padding), mode='constant', value=0)

    # Convolve with the Gaussian kernel
    gauss_kernel = gaussian_kernel(kernel_size, sigma)
    gauss_kernel = gauss_kernel.expand(input_tensor.shape[1], 1, kernel_size, kernel_size).type_as(input_tensor)
    smoothed_tensor = F.conv2d(padded_tensor, gauss_kernel, stride=1, padding=0, groups=input_tensor.shape[1])

    # Remove padding if necessary
    final_tensor = smoothed_tensor[:, :, padding:-padding, padding:-padding]
    final_tensor = F.interpolate(final_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)

    return final_tensor