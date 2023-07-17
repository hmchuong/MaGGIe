from torch.nn import functional as F

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
        x = F.interpolate(x, size=size, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    x = x.view(*shape[:-2], *x.shape[-2:]).to(dtype)
    return x
