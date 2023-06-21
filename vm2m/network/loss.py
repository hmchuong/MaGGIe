import torch
import torch.nn as nn
from torch.nn import functional as F

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