from skimage.measure import label
import numpy as np
import cv2
import torch.nn.functional as F

from .metric import reshape2D

def reverse_transform(img, transform_info):
    '''
    img: bs, ..., h, w

    '''
    img_shape = list(img.shape)
    img = reshape2D(img)
    for transform in transform_info[::-1]:
        
        name = transform['name'][0]
        if name == 'padding':
            pad_h, pad_w  = transform['pad_size']
            pad_h, pad_w = pad_h.item(), pad_w.item()
            h, w = img.shape[-2:]
            img = img[:, :h-pad_h, :w-pad_w]

        elif name == 'resize':
            h, w = transform['ori_size']
            h, w = h.item(), w.item()
            img = [cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR) for img in img]
            img = np.stack(img, axis=0)
            img_shape[-2:] = h, w

    img = img.reshape(img_shape)

    return img

def reverse_transform_tensor(img, transform_info):
    '''
    img: bs, ..., h, w

    '''
    img_shape = list(img.shape)
    img = reshape2D(img)
    for transform in transform_info[::-1]:
        
        name = transform['name'][0]
        if name == 'padding':
            pad_h, pad_w  = transform['pad_size']
            pad_h, pad_w = pad_h.item(), pad_w.item()
            h, w = img.shape[-2:]
            img = img[:, :h-pad_h, :w-pad_w]

        elif name == 'resize':
            h, w = transform['ori_size']
            h, w = h.item(), w.item()
            # img = [cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR) for img in img]
            # img = np.stack(img, axis=0)
            img = F.interpolate(img.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=True).squeeze(1)
            img_shape[-2:] = h, w

    img = img.reshape(img_shape)

    return img

def _postprocess(alpha, orih=None, oriw=None, bbox=None):
    labels=label((alpha>0.05).astype(int))
    try:
        assert( labels.max() != 0 )
    except:
        return alpha
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    alpha = alpha * largestCC
    if bbox is None:
        return alpha
    else:
        ori_alpha = np.zeros(shape=[orih, oriw], dtype=np.float32)
        ori_alpha[bbox[0]:bbox[1], bbox[2]:bbox[3]] = alpha
        return ori_alpha

def postprocess(alpha):
    alpha_shape = alpha.shape
    alpha = reshape2D(alpha) 
    alpha = [_postprocess(a) for a in alpha]
    alpha = np.stack(alpha, axis=0)
    alpha = alpha.reshape(alpha_shape)
    return alpha