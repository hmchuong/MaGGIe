import torch
import torch.nn as nn
from torch.nn import functional as F
# from pudb.remote import set_trace

def _loss_dtSSD(pred, gt, mask):
    b, n_f, _, h, w = pred.shape
    dadt = pred[:, 1:] - pred[:, :-1]
    dgdt = gt[:, 1:] - gt[:, :-1]
    # import pdb; pdb.set_trace()
    diff = (dadt - dgdt) ** 2
    diff = diff * mask[:, 1:]
    # import pdb; pdb.set_trace()
    diff = torch.sum(diff) / torch.sum(mask[:, 1:] + 1e-6)
    return diff

def _loss_dtSSD_ohem_smooth_l1(pred, gt, mask, ratio=0.5, threshold=5e-5):
    '''
    ratio: maximum ratio of hard examples
    threshold: minimum loss of hard examples
    '''
    dadt = torch.abs(pred[:, 1:] - pred[:, :-1])
    dgdt = torch.abs(gt[:, 1:] - gt[:, :-1])
    diff = F.smooth_l1_loss(dadt, dgdt, reduction='none', beta=0.5)
    diff = diff * mask[:, 1:]
    diff = diff[mask[:, 1:] > 0]
    
    # Ohem
    diff, _ = torch.sort(diff, descending=True)
    
    min_value = diff[int(diff.numel() * ratio)]
    min_value = max(min_value, threshold)
    min_value = min(min_value, diff.max())

    hard_loss = diff[diff > min_value].mean()
    return hard_loss

def loss_dtSSD(pred, gt, mask):
    loss = _loss_dtSSD(pred, gt, mask) #+ _loss_dtSSD(torch.flip(pred, dims=(1,)), torch.flip(gt, dims=(1,)), torch.flip(mask, dims=(1,)))
    # loss = _loss_dtSSD_ohem_smooth_l1(pred, gt, mask)
    return loss
    # b, n_f, _, h, w = pred.shape
    # import pdb; pdb.set_trace()
    # dadt = pred[:, 1:] - pred[:, :-1]
    # dgdt = gt[:, 1:] - gt[:, :-1]
    # diff = (dadt - dgdt) ** 2
    # diff = diff * mask[:, 1:]
    # # import pdb; pdb.set_trace()
    # diff = torch.sum(diff) / torch.sum(mask[:, 1:])
    # return diff
    # metric = torch.sqrt(torch.sum((dadt - dgdt) ** 2, dim=(2, 3, 4)))
    # metric = torch.sum(metric)
    # count = ((n_f - 1) * b)
    # if torch.isnan(metric).any():
    #     # set_trace()
    # return metric/ (count + 1e-4)

def loss_comp(alpha_pred, alpha_gt, fg, bg, mask):
    comp_pred = alpha_pred * fg + (1 - alpha_pred) * bg
    comp_gt = alpha_gt * fg + (1 - alpha_gt) * bg
    loss = torch.sum(F.l1_loss(comp_pred, comp_gt, reduction='none')) / (mask.sum() + 1e-6)
    return loss

class GradientLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.kernel_x, self.kernel_y = self.sobel_kernel()
        self.eps = eps

    def forward(self, logit, label, mask=None):
        if len(label.shape) == 3:
            label = label.unsqueeze(1)
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
            logit = logit * mask
            label = label * mask
            # import pdb; pdb.set_trace()
            loss = torch.sum(
                F.l1_loss(self.sobel(logit), self.sobel(label), reduction='none')) / (
                    mask.sum() + self.eps)
        else:
            loss = F.l1_loss(self.sobel(logit), self.sobel(label), 'mean')

        return loss

    def sobel(self, input):
        """Using Sobel to compute gradient. Return the magnitude."""
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect NCHW, but it is ",
                             input.shape)

        n, c, h, w = input.shape
        
        self.kernel_x = self.kernel_x.to(input.device)
        self.kernel_y = self.kernel_y.to(input.device)

        input_pad = input.reshape(n * c, 1, h, w)
        input_pad = F.pad(input_pad, pad=[1, 1, 1, 1], mode='replicate')
        grad_x = F.conv2d(input_pad, self.kernel_x, padding=0)
        grad_y = F.conv2d(input_pad, self.kernel_y, padding=0)

        mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)
        mag = mag.reshape(n, c, h, w)

        return mag

    def sobel_kernel(self):
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0],
                                     [-1.0, 0.0, 1.0]]).float()
        kernel_x = kernel_x / kernel_x.abs().sum()
        kernel_y = kernel_x.permute(1, 0)
        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)
        return kernel_x, kernel_y

def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel.to(img.device), groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

def weight_pyramid(x, max_levels=3):
    current = x
    pyr = []
    for level in range(max_levels):
        down = downsample(current)
        pyr.append(current)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)

    def l1_loss(self, input, target, weight=None):
        if weight is None:
            return F.l1_loss(input, target)
        else:
            return (F.l1_loss(input, target, reduction='none') * weight).sum() / (weight.sum() + 1e-6)
        
    def forward(self, input, target, weight=None):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        weights = [None] * len(pyr_input)
        if weight is not None:
            weights = weight_pyramid(weight, max_levels=self.max_levels)
        
        total_loss = 0
        for i in range(len(pyr_input)):
            total_loss += self.l1_loss(pyr_input[i], pyr_target[i], weights[i])
        return total_loss

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y ):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss