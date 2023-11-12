import os
import random
import math
from typing import Any
import numpy as np
import cv2
import torch
from PIL import Image
import albumentations as A
import imgaug.augmenters as iaa
from imgaug import parameters as iap
from skimage import exposure

try:
    from .utils import random_transform
except ImportError:
    from utils import random_transform

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_dict: dict):
        transform_info = []
        input_dict["transform_info"] = transform_info
        for t in self.transforms:
            input_dict = t(input_dict)
            # import pdb; pdb.set_trace()
        return input_dict

class ChooseOne(object):
    def __init__(self, random, transforms):
        self.random = random
        self.transforms = transforms

    def __call__(self, input_dict: dict):
        t = self.random.choice(self.transforms)
        input_dict = t(input_dict)
        return input_dict

class Load(object):
    def __init__(self, is_rgb=True):
        self.is_rgb = is_rgb
    def __call__(self, input_dict: dict):
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]
        # weights = input_dict["weights"]

        frames = [np.array(Image.open(frame_path).convert("RGB" if self.is_rgb else "BGR")) for frame_path in frames]
        if masks is not None:
            masks = [np.array(Image.open(mask_path).convert("L")) for mask_path in masks]
        alphas = [np.array(Image.open(alpha_path).convert("L")) for alpha_path in alphas]
        # if weights is not None:
        #     loaded_weights = []
        #     for weight_path in weights:
        #         if not os.path.exists(weight_path):
        #             weight = np.ones_like(alphas[0]) * 255
        #         else:
        #             weight = np.array(Image.open(weight_path).convert("L"))
        #         weight = ((weight > 127) * 255).astype('uint8')
        #         loaded_weights.append(weight)
        #     weights = loaded_weights
       
        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        input_dict["masks"] = masks
        # input_dict["weights"] = weights
        return input_dict

class RandomCenterCrop(object):
    def __init__(self, random) -> None:
        '''
        Random crop to retain 1/4 center of the image then resize to output size
        '''
        self.random = random
    
    def __call__(self, input_dict: dict) -> Any:
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]
        # weights = input_dict["weights"]

        h, w = frames[0].shape[:2]
        
        margin_h = int(h * 0.25) + self.random.randint(0, int(h * 0.25))
        margin_w  = int(w * 0.25) + self.random.randint(0, int(w * 0.25))
        x = h // 2 - margin_h 
        y = w // 2 - margin_w
        new_w = margin_w * 2
        new_h = margin_h * 2

        frames = [frame[y:y+new_h, x:x+new_w, :] for frame in frames]
        if masks is not None:
            masks = [mask[y:y+new_h, x:x+new_w] for mask in masks]
        # if weights is not None:
        #     weights = [weight[y:y+new_h, x:x+new_w] for weight in weights]
        alphas = [alpha[y:y+new_h, x:x+new_w] for alpha in alphas]
        
        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        input_dict["masks"] = masks
        # input_dict["weights"] = weights
        
        return input_dict

class ResizeShort(object):
    def __init__(self, short_size, transform_alphas=True):
        self.short_size = short_size
        self.transform_alphas = transform_alphas
    
    def __call__(self, input_dict: dict):
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]
        # weights = input_dict["weights"]

        input_dict["ori_alphas"] = alphas
        transform_info = input_dict["transform_info"]
        h, w = frames[0].shape[:2]
        ratio = self.short_size * 1.0 / min(w, h) 
        if ratio != 1:
            frames = [cv2.resize(frame, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR) for frame in frames]
            if masks is not None:
                masks = [cv2.resize(mask, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST) for mask in masks]
            # if weights is not None:
            #     weights = [cv2.resize(mask, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST) for mask in weights]
            # import pdb; pdb.set_trace()
            alphas = [cv2.resize(alpha, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR) for alpha in alphas]
        transform_info.append({'name': 'resize', 'ori_size': (h, w), 'ratio': ratio})
        
        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        input_dict["masks"] = masks
        # input_dict["weights"] = weights
        input_dict["transform_info"] = transform_info
        
        return input_dict

class PaddingMultiplyBy(object):
    def __init__(self, divisor=32, transform_alphas=True):
        self.divisor = divisor
        self.transform_alphas = transform_alphas

    def __call__(self, input_dict: dict):
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]
        # weights = input_dict["weights"]
        transform_info = input_dict["transform_info"]

        h, w = frames[0].shape[:2]
        h_pad = (self.divisor - h % self.divisor) % self.divisor
        w_pad = (self.divisor - w % self.divisor) % self.divisor
        frames = [cv2.copyMakeBorder(frame, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0) for frame in frames]
        if masks is not None:
            masks = [cv2.copyMakeBorder(mask, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0) for mask in masks]
        # if weights is not None:
        #     weights = [cv2.copyMakeBorder(mask, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0) for mask in weights]
        alphas = [cv2.copyMakeBorder(alpha, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0) for alpha in alphas]
        transform_info.append({'name': 'padding', 'pad_size': (h_pad, w_pad)})
        
        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        input_dict["masks"] = masks
        # input_dict["weights"] = weights
        input_dict["transform_info"] = transform_info

        return input_dict

class Stack(object):
    def __init__(self):
        pass
    def __call__(self, input_dict: dict):
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]
        # weights = input_dict["weights"]

        frames = np.stack(frames, axis=0)
        alphas = np.stack(alphas, axis=0)
        if masks is not None:
            masks = np.stack(masks, axis=0)
        # if weights is not None:
        #     weights = np.stack(weights, axis=0)
        
        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        input_dict["masks"] = masks
        # input_dict["weights"] = weights

        return input_dict

class RandomCropByAlpha(object):
    def __init__(self, crop_size, random, padding_prob=0.5):
        self.crop_size = crop_size
        self.random = random
        self.padding_prob = padding_prob
    
    def crop(self, frames, alphas, masks, min_x, min_y, max_x, max_y, w, h):
        max_x = max(max_x - self.crop_size[1], min_x + 1)
        max_y = max(max_y - self.crop_size[0], min_y + 1)

        for _ in range(3):
            x, y = self.random.randint(min_x, max_x), self.random.randint(min_y, max_y)

            x = min(x, w - self.crop_size[1])
            y = min(y, h - self.crop_size[0])

            crop_frames = frames[:, y:y+self.crop_size[0], x:x+self.crop_size[1], :]
            crop_alphas = alphas[:, y:y+self.crop_size[0], x:x+self.crop_size[1]]
            if (crop_alphas > 127).sum() > 0:
                break

        crop_masks = masks
        if masks is not None:
            crop_masks = masks[:,y:y+self.crop_size[0], x:x+self.crop_size[1]]
        return crop_frames, crop_alphas, crop_masks
    
    def __call__(self, input_dict: dict):
        '''
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) or None
        '''

        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]
        # weights = input_dict["weights"]

        h, w = frames[0].shape[:2]
        if h < self.crop_size[0] or w < self.crop_size[1]:
            raise ValueError("Crop size {} is larger than image size {}".format(self.crop_size, (h, w)))
        
        # Find alpha region
        try:
            ys, xs = np.where(alphas.mean(0) > 127)
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
        except:
            min_x, max_x = 0, w
            min_y, max_y = 0, h

        if self.random.rand() > self.padding_prob:
            # Cropping
            crop_frames, crop_alphas, crop_masks = self.crop(frames, alphas, masks, min_x, min_y, max_x, max_y, w, h)
        else:
            # Padding and resize to crop size
            if h > w:
                pad_w = (h - w) // 2
                pad_h = 0
            else:
                pad_w = 0
                pad_h = (w - h) // 2
            crop_frames = [cv2.copyMakeBorder(frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0) for frame in frames]
            crop_alphas = [cv2.copyMakeBorder(alpha, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0) for alpha in alphas]
            crop_frames = [cv2.resize(frame, self.crop_size, interpolation=cv2.INTER_LINEAR) for frame in crop_frames]
            crop_alphas = [cv2.resize(alpha, self.crop_size, interpolation=cv2.INTER_LINEAR) for alpha in crop_alphas]
            crop_frames = np.stack(crop_frames, axis=0)
            crop_alphas = np.stack(crop_alphas, axis=0)
            if masks is not None:
                crop_masks = [cv2.copyMakeBorder(mask, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0) for mask in masks]
                crop_masks = np.stack([cv2.resize(mask, self.crop_size, interpolation=cv2.INTER_NEAREST) for mask in crop_masks], axis=0)
            else:
                crop_masks = None
        
        # crop_weights = weights
        # if weights is not None:
        #     crop_weights = weights[:,y:y+self.crop_size[0], x:x+self.crop_size[1]]
        
        input_dict["frames"] = crop_frames
        input_dict["alphas"] = crop_alphas
        input_dict["masks"] = crop_masks
        # input_dict["weights"] = crop_weights

        return input_dict

class RandomHorizontalFlip(object):
    def __init__(self, random, p=0.5):
        self.random = random
        self.p = p

    def __call__(self, input_dict: dict):
        '''
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) or None
        '''
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]
        # weights = input_dict["weights"]

        if self.random.rand() < self.p:
            frames = frames[:, :, ::-1, :]
            alphas = alphas[:, :, ::-1]
            if masks is not None:
                masks = masks[:, :, ::-1]
            # if weights is not None:
            #     weights = weights[:, :, ::-1]
        
        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        input_dict["masks"] = masks
        # input_dict["weights"] = weights

        return input_dict

class LoadRandomBackground(object):
    def __init__(self, bg_paths, random, is_rgb=True, blur_p=0.5, blur_kernel_size=[5, 15, 25], blur_sigma=[1.0, 1.5, 3.0, 5.0]):
        self.bg_paths = bg_paths
        self.random = random
        self.is_rgb = is_rgb
        self.blur_p = blur_p
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
    
    def __call__(self, input_dict: dict):
        '''
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) or None
        '''
        frames = input_dict["frames"]

        bg_path = self.random.choice(self.bg_paths)
        bg = cv2.imread(bg_path)
        if self.is_rgb:
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        
        # Blur background
        if self.random.rand() < self.blur_p:
            blur_ks = self.random.choice(self.blur_kernel_size)
            blur_sigma = self.random.choice(self.blur_sigma)
            bg = cv2.GaussianBlur(bg, (blur_ks, blur_ks), blur_sigma)
        
        # Crop background following frames size
        h, w = frames[0].shape[:2]
        bh, bw = bg.shape[:2]
        x = self.random.randint(0, bw - w)
        y = self.random.randint(0, bh - h)
        bg = bg[y:y+h, x:x+w, :]
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)

        # Compose background
        frames = frames.astype(np.float32)
        bg = bg.astype(np.float32)

        input_dict["fg"] = frames
        input_dict["bg"] = np.tile(bg, (frames.shape[0], 1, 1, 1))
        
        return input_dict

class ComposeBackground(object):
    def __call__(self, input_dict: dict):
        '''
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) or None
        '''
        alphas = input_dict["alphas"]
        bg = input_dict["bg"]
        fg = input_dict["fg"]

        # Compose background
        fg = fg.astype(np.float32)
        bg = bg.astype(np.float32)
        compose_frames = fg * (alphas[..., None].astype(float) / 255.0) + bg * (1 - alphas[..., None].astype(float) / 255.0)
        compose_frames = np.clip(compose_frames, 0, 255).astype(np.uint8)
        input_dict["frames"] = compose_frames
        
        return input_dict

class MasksFromBinarizedAlpha(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def __call__(self, input_dict: dict):
        
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]

        if masks is None:
            masks = [(a > self.threshold * 255).astype(np.uint8) * 255 for a in alphas]
        
        input_dict["masks"] = masks

        return input_dict
    
class RandomBinarizeAlpha(object):
    def __init__(self, random, binarize_max_k=30):
        self.random = random
        self.binaraize_max_k = binarize_max_k

    def _gen_single_mask(self, alpha):
        '''
        alpha: (H, W)
        '''
        
        threshold = self.random.uniform(0.1, 0.95) * 255

        # binarize the alphas
        _, img_binarized = cv2.threshold(np.array(alpha), threshold, 1, cv2.THRESH_BINARY)

        # generate random kernel sizes for dilation and erosion
        kernel_size_dilate = self.random.randint(1, self.binaraize_max_k)
        kernel_size_erode = self.random.randint(1, self.binaraize_max_k)

        # create the kernels
        kernel_dilate = np.ones((kernel_size_dilate, kernel_size_dilate), np.uint8)
        kernel_erode = np.ones((kernel_size_erode, kernel_size_erode), np.uint8)

        # randomly decide the order of dilation and erosion and whether to do both or just one
        operation_order = self.random.choice(["dilate_erode", "erode_dilate", "dilate", "erode"])

        if operation_order == "dilate_erode":
            img_dilated = cv2.dilate(img_binarized, kernel_dilate, iterations=1)
            img_final = cv2.erode(img_dilated, kernel_erode, iterations=1)
        elif operation_order == "erode_dilate":
            img_eroded = cv2.erode(img_binarized, kernel_erode, iterations=1)
            img_final = cv2.dilate(img_eroded, kernel_dilate, iterations=1)
        elif operation_order == "dilate":
            img_final = cv2.dilate(img_binarized, kernel_dilate, iterations=1)
        else:  # operation_order == "erode"
            img_final = cv2.erode(img_binarized, kernel_erode, iterations=1)
        return img_final * 255

    def __call__(self, input_dict: dict):
        '''
        Args:
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) or None
        
        Args:
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) from alphas
        '''
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]
        alphas[alphas < 5] = 0
        if masks is None:
            masks = np.stack([self._gen_single_mask(alpha) for alpha in alphas], axis=0)
        
        input_dict["masks"] = masks

        return input_dict

class RandomBinarizedMask(RandomBinarizeAlpha):
    def __call__(self, input_dict: dict):
        '''
        Args:
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) or None
        
        Args:
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) from alphas
        '''
        masks = input_dict["masks"]
        
        # n_inst = len(masks) // len(frames) 
        # The same mask augmentation for the same instance
        # for i in range(n_inst):
        #     alpha = masks[i::n_inst].transpose(1, 2, 0)
        #     masks[i::n_inst] = self._gen_single_mask(alpha).transpose(2, 0, 1)
        
        input_dict["masks"] = np.stack([self._gen_single_mask(alpha) for alpha in masks], axis=0)
        return input_dict

class GenMaskFromAlpha(object):
    def __init__(self, threshold=0.5):
        self.threshold = 0.5
        pass

    def __call__(self, input_dict: dict):
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]
        h, w = alphas.shape[-2:]
        new_masks = ((alphas > 127) * 255).astype('uint8')
        input_dict["masks"] = np.stack([cv2.resize(m, (w, h), cv2.INTER_NEAREST) for m in new_masks], axis=0)

        return input_dict
    
class DownUpMask(object):
    def __init__(self, random, ratio, p=0.5):
        self.random = random
        self.ratio = ratio
        self.p = p

    def downup(self, mask):
        if self.random.rand() < self.p:
            h, w = mask.shape[:2]
            mask = cv2.resize(mask, (0, 0), fx=self.ratio, fy=self.ratio, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 127).astype('uint8') * 255
        return mask
    def __call__(self, input_dict: dict):
        masks = input_dict["masks"]
        input_dict["masks"] = np.stack([self.downup(m) for m in masks], axis=0)

        return input_dict

class CutMask(object):
    def __init__(self, random):
        self.internal_perturb_prob = 0.5
        self.external_perturb_prob = 0.5
        self.random = random

    def internal(self, mask):
        if self.random.rand() < self.internal_perturb_prob:
            h, w = mask.shape
            perturb_size_h, perturb_size_w = self.random.randint(h // 8, h // 4), self.random.randint(w // 8, w // 4)
            x = self.random.randint(0, h - perturb_size_h)
            y = self.random.randint(0, w - perturb_size_w)
            x1 = self.random.randint(0, h - perturb_size_h)
            y1 = self.random.randint(0, w - perturb_size_w)
            mask[x:x+perturb_size_h, y:y+perturb_size_w] = mask[x1:x1+perturb_size_h, y1:y1+perturb_size_w].copy()
        return mask

    def external(self, mask):
        if self.random.rand() < self.external_perturb_prob and mask.shape[0] > 1:
            i, j = random.sample(list(range(0, mask.shape[0])), k=2)
            h, w = mask.shape[-2:]
            perturb_size_h, perturb_size_w = self.random.randint(h // 8, h // 4), self.random.randint(w // 8, w // 4)
            x = self.random.randint(0, h - perturb_size_h)
            y = self.random.randint(0, w - perturb_size_w)
            mask_i_perturb = mask[i, x:x+perturb_size_h, y:y+perturb_size_w].copy()
            mask_j_perturb = mask[j, x:x+perturb_size_h, y:y+perturb_size_w].copy()
            mask[i, x:x+perturb_size_h, y:y+perturb_size_w] = mask_j_perturb
            mask[j, x:x+perturb_size_h, y:y+perturb_size_w] = mask_i_perturb
        return mask

    def __call__(self, sample: dict):
        if self.random.random() < 0.5:
            sample['masks'] = np.stack([self.internal(sample['masks'][i]) for i in range(sample['masks'].shape[0])], axis=0)
        else:
            sample['masks'] = self.external(sample['masks'])
        return sample

class MaskDropout(object):
    def __init__(self, random):
        self.random = random  
    
    def __call__(self, sample):
        ''' Drop a region in center of the mask
        '''
        masks = sample['masks']
        if self.random.rand() < 0.5 or masks.shape[0] // 2 < 3:
            return sample
        
        n_dropouts = self.random.randint(1, masks.shape[0] // 2)
        selected_indices = self.random.choice(masks.shape[0], n_dropouts, replace=False)
        for i in selected_indices:
            ys, xs = np.where(masks[i] > 0)
            if len(ys) == 0:
                continue
            xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
            if (ymax - ymin + 1) // 8 < 2 or (xmax - xmin + 1) // 8 < 2:
                continue
            
            perturb_size_h, perturb_size_w = self.random.randint((ymax - ymin + 1) // 16, (ymax - ymin + 1) // 8), self.random.randint((xmax - xmin + 1) // 16, (xmax - xmin + 1) // 8)
            idx = self.random.choice(range(len(ys)), 1)
            x, y = int(xs[idx]), int(ys[idx])
            
            x = min(x, xmax - perturb_size_w)
            y = min(y, ymax - perturb_size_h)
            masks[i, y:y+perturb_size_h, x:x+perturb_size_w] = 0
        sample['masks'] = masks
        return sample

def get_random_structure(size):
    # The provided model is trained with 
    #   choice = np.random.randint(4)
    # instead, which is a bug that we fixed here
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)

def perturb_seg(gt, iou_target=0.6):
    h, w = gt.shape
    seg = gt.copy()

    _, seg = cv2.threshold(seg, 127, 255, 0)

    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.25:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])

        if compute_iou(seg, gt) < iou_target:
            break

    return seg

class ModifyMaskBoundary(object):
    def __init__(self, random, p=0.5, regional_sample_rate=0.1, sample_rate=0.1, move_rate=0.0):
        self.random = random
        self.p = p
        self.regional_sample_rate = regional_sample_rate
        self.sample_rate = sample_rate
        self.move_rate = move_rate
    
    def modify_mask(self, image):
        if self.random.rand() < self.p:
            return image
            
        iou_target = self.random.rand() * 0.2 + 0.8

        if int(cv2.__version__[0]) >= 4:
            contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        #only modified contours is needed actually. 
        sampled_contours = []   
        modified_contours = [] 

        for contour in contours:
            if contour.shape[0] < 10:
                continue
            M = cv2.moments(contour)

            #remove region of contour
            number_of_vertices = contour.shape[0]
            number_of_removes = int(number_of_vertices * self.regional_sample_rate)
            
            idx_dist = []
            for i in range(number_of_vertices - number_of_removes):
                idx_dist.append([i, np.sum((contour[i] - contour[i+number_of_removes])**2)])
                
            idx_dist = sorted(idx_dist, key=lambda x:x[1])
            candidates = idx_dist[:math.ceil(0.1*len(idx_dist))]
            remove_start = candidates[self.random.choice(np.arange(len(candidates)))][0]
            
            new_contour = np.concatenate([contour[:remove_start], contour[remove_start+number_of_removes:]], axis=0)
            contour = new_contour
            

            #sample contours
            number_of_vertices = contour.shape[0]
            indices = self.random.choice(range(number_of_vertices), int(number_of_vertices * self.sample_rate), replace=False)
            indices.sort()
            sampled_contour = contour[indices]
            sampled_contours.append(sampled_contour)

            modified_contour = np.copy(sampled_contour)
            if (M['m00'] != 0):
                center = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])

                #modify contours
                for idx, coor in enumerate(modified_contour):

                    change = np.random.normal(0, self.move_rate) # 0.1 means change position of vertex to 10 percent farther from center
                    x,y = coor[0]
                    new_x = x + (x-center[0]) * change
                    new_y = y + (y-center[1]) * change

                    modified_contour[idx] = [new_x,new_y]
            modified_contours.append(modified_contour)
            

        #draw boundary
        gt = np.copy(image)
        image = np.zeros_like(image)

        modified_contours = [cont for cont in modified_contours if len(cont) > 0]
        if len(modified_contours) == 0:
            image = gt.copy()
        else:
            image = cv2.drawContours(image, modified_contours, -1, (255, 0, 0), -1)

        image = perturb_seg(image, iou_target)
        
        return image

    def __call__(self, input_dict: dict):
        masks = input_dict["masks"]
        input_dict["masks"] = np.stack([self.modify_mask(m) for m in masks], axis=0)

        return input_dict


class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self, input_dict: dict):
        '''
        Args:
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) or None

        Returns:
        frames: (T, C, H, W)
        alphas: (T, n_ints, H, W)
        masks: (T, n_ints, H, W) or None
        '''
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]
        # weights = input_dict["weights"]

        frames = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).contiguous().float()        
        alphas = torch.from_numpy(np.ascontiguousarray(alphas)).contiguous()
        n_insts = alphas.shape[0] // frames.shape[0]
        alphas = alphas.view(frames.shape[0], n_insts, *alphas.shape[1:])
        alphas[alphas < 5] = 0

        if masks is not None:
            masks = torch.from_numpy(np.ascontiguousarray(masks).astype('uint8')).contiguous()
            masks = masks.view(frames.shape[0], n_insts, *masks.shape[1:])
        
        # if weights is not None:
        #     weights = torch.from_numpy(np.ascontiguousarray(weights).astype('uint8')).contiguous()
        #     weights = weights.view(frames.shape[0], n_insts, *weights.shape[1:])


        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        input_dict["masks"] = masks
        # input_dict["weights"] = weights
        
        if "ori_alphas" in input_dict:
            ori_alphas = input_dict["ori_alphas"]
            ori_alphas = torch.from_numpy(np.ascontiguousarray(ori_alphas)).contiguous()
            ori_alphas = ori_alphas.view(frames.shape[0], n_insts, *ori_alphas.shape[1:])
            input_dict["ori_alphas"] = ori_alphas
        
        if "fg" in input_dict:
            input_dict["fg"] = torch.from_numpy(np.ascontiguousarray(input_dict["fg"])).permute(0, 3, 1, 2).contiguous().float()
        if "bg" in input_dict:
            input_dict["bg"] = torch.from_numpy(np.ascontiguousarray(input_dict["bg"])).permute(0, 3, 1, 2).contiguous().float()
        return input_dict

class Normalize(object):
    def __init__(self, mean=[], std=[]):
        self.mean = mean
        self.std = std
    
    def norm(self, frames):
        mean = torch.tensor(self.mean).view(1, 3, 1, 1).float()
        std = torch.tensor(self.std).view(1, 3, 1, 1).float()
        frames = frames / 255.0
        frames = (frames - mean) / std
        return frames
    
    def __call__(self, input_dict: dict):
        frames = input_dict["frames"]    
        input_dict["frames"] = self.norm(frames)
        
        # TODO: Compute FG, BG based on alpha
        alphas = input_dict["alphas"] / 255.0
        alphas = alphas[:, :, None]
        fg_shape = (*alphas.shape[:2], 3, *alphas.shape[-2:]) 

        norm_frames = frames[:, None].expand(fg_shape) / 255.0
        fg = norm_frames / alphas
        fg = torch.nan_to_num(fg, nan=0.0, posinf=0.0)
        fg = torch.clamp(fg, 0, 1) # T x N_i x 3 x H x W
        bg = norm_frames - fg * alphas # # T x N_i x 3 x H x W
        bg = bg / (1 - alphas)
        bg = torch.nan_to_num(bg, nan=0.0)
        bg = torch.clamp(bg, 0, 1)

        if "fg" in input_dict:
            input_dict["fg"] = self.norm(input_dict["fg"])
        else:
            input_dict["fg"] = fg
        if "bg" in input_dict:
            input_dict["bg"] = self.norm(input_dict["bg"])
        else:
            input_dict["bg"] = bg
        return input_dict

class GammaContrast(object):
    def __init__(self, random, gamma=(1.0, 0.2, 0.5, 1.5), p=0.3):
        self.pixel_aug_gamma = iaa.GammaContrast(gamma=iap.TruncatedNormal(*gamma))
        self.p = p
        self.random = random

    def __call__(self, input_dict: dict):
        if self.random.rand() > self.p:
            return input_dict
        
        # Augment frames and fg
        frames = input_dict["frames"]
        aug = self.pixel_aug_gamma.to_deterministic()
        for i in range(frames.shape[0]):
            frames[i] = aug.augment_image(frames[i])
        input_dict["frames"] = frames

        if "fg" in input_dict:
            input_dict["fg"] = frames
        
        if "bg" in input_dict:
            bg = input_dict["bg"]
            aug = self.pixel_aug_gamma.to_deterministic()
            for i in range(bg.shape[0]):
                bg[i] = aug.augment_image(bg[i])
            input_dict["bg"] = bg

        return input_dict

class HistogramMatching(object):
    def __init__(self, random, p=0.3):
        self.random = random
        self.p = p
    
    def __call__(self, input_dict: dict):
        if "bg" not in input_dict or self.random.rand() > self.p:
            return input_dict

        fg = input_dict["fg"].astype(np.float32)
        bg = input_dict["bg"].astype(np.float32)
        ratio = self.random.uniform(0, 0.5)
        if self.random.rand() < 0.05:
            bg_match = exposure.match_histograms(bg, fg, channel_axis=-1)
            bg = bg_match * ratio + bg * (1. - ratio)
        else:
            fg_match = exposure.match_histograms(fg, bg, channel_axis=-1)
            fg = fg_match * ratio + fg * (1. - ratio)
        fg = fg.astype(np.uint8)
        bg = bg.astype(np.uint8)
        input_dict["fg"] = fg
        input_dict["frames"] = fg
        input_dict["bg"] = bg
        return input_dict

class AdditiveGaussionNoise(object):
    def __init__(self, random, p=0.3):
        self.random = random
        self.p = p
        self.pixel_aug_gaussian = iaa.AdditiveGaussianNoise(scale=(0, 0.03*255))
    
    def __call__(self, input_dict: dict):
        if self.random.rand() > self.p:
            return input_dict
        frames = input_dict["frames"]
        fg = input_dict.get("fg", None)
        bg = input_dict.get("bg", None)
        aug = self.pixel_aug_gaussian.to_deterministic()
        for i in range(frames.shape[0]):
            frames[i] = aug.augment_image(np.uint8(frames[i]))
            if fg is not None:
                fg[i] = frames[i]
            if bg is not None:
                bg[i] = aug.augment_image(np.uint8(bg[i]))

        input_dict["frames"] = frames
        if fg is not None:
            input_dict["fg"] = fg
        if bg is not None:
            input_dict["bg"] = bg
        
        return input_dict

class JpegCompression(object):
    def __init__(self, random, p=0.3):
        self.random = random
        self.p = p
        self.jpeg_aug = iaa.JpegCompression(compression=(20, 80)) 
    
    def __call__(self, input_dict: dict):
        if self.random.rand() > self.p:
            return input_dict
        frames = input_dict["frames"]
        fg = input_dict.get("fg", None)
        bg = input_dict.get("bg", None)
        # alphas = input_dict.get("alphas", None)

        aug = self.jpeg_aug.to_deterministic()
        for i in range(frames.shape[0]):
            frames[i] = aug.augment_image(np.uint8(frames[i]))

            if fg is not None:
                fg[i] = frames[i]
            if bg is not None:
                bg[i] = aug.augment_image(np.uint8(bg[i]))
        # for i in range(alphas.shape[0]):
        #     alphas[i] = aug.augment_image(np.uint8(alphas[i]))
        input_dict["frames"] = frames
        # input_dict["alphas"] = alphas
        if fg is not None:
            input_dict["fg"] = fg
        if bg is not None:
            input_dict["bg"] = bg
        return input_dict
    
class RandomAffine(object):
    def __init__(self, random, p=0.5):
        self.random = random
        self.p = p
    
    def __call__(self, input_dict: dict):
        if self.random.rand() > self.p:
            return input_dict
        frames = input_dict["frames"]
        bg = input_dict.get("bg", None)
        alphas = input_dict.get("alphas", None)
        # weights = input_dict.get("weights", None)

        ignore_regions = np.ones_like(alphas)
        # list_FM = list(frames) + list(alphas) + list(weights) + list(ignore_regions)
        list_FM = list(frames) + list(alphas) + list(ignore_regions)
        if bg is not None:
            list_FM += list(bg)
        
        list_trans_FM = random_transform(list_FM, self.random, rt=10, sh=5, zm=[0.95,1.05], sc= [1, 1], cs=0.03*255., hf=False)
        n_f = len(frames)
        n_alpha = len(alphas)
        frames = np.stack(list_trans_FM[:n_f], axis=0)
        alphas = np.stack(list_trans_FM[n_f: n_f + n_alpha], axis=0)
        # weights = np.stack(list_trans_FM[n_f + n_alpha: n_f + n_alpha * 2], axis=0)
        # ignore_regions = np.stack(list_trans_FM[n_f + n_alpha * 2: 2*n_f + n_alpha * 2], axis=0)
        ignore_regions = np.stack(list_trans_FM[n_f + n_alpha: n_f + n_alpha * 2], axis=0)
        if bg is not None:
            bg = np.stack(list_trans_FM[3*n_f:], axis=0)
        
        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        # input_dict["weights"] = weights
        input_dict["ignore_regions"] = ignore_regions
        if bg is not None:
            input_dict["bg"] = bg
            input_dict["fg"] = frames
        return input_dict

class MotionBlur(object):
    def __init__(self, random, p=0.3) -> None:
        self.random = random
        self.p = p
        self.motion_aug = A.MotionBlur(p=1.0, blur_limit=(3,49))
    
    def __call__(self, input_dict: dict):
        if self.random.rand() > self.p:
            return input_dict

        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        # weights = input_dict["weights"]
        bg = input_dict.get("bg", None)
        
        if self.random.uniform(0, 1) < 0.5 and bg is not None:
            N_cat = np.concatenate([frames, bg, alphas[:, :, :, None], weights[:, :, :, None]], axis=-1) # T x H x W x 7
            N_cat = N_cat.transpose((1, 2, 3, 0)) # H x W x 7 x T
            N_cat = N_cat.reshape(*N_cat.shape[:2], -1) # H x W x 7T
            N_cat_aug = self.motion_aug(image=N_cat)["image"] # H x W x 7T
            N_cat_aug = N_cat_aug.reshape(*N_cat_aug.shape[:2], -1, frames.shape[0]) # H x W x 7 x T
            N_cat_aug = N_cat_aug.transpose((3, 0, 1, 2)) # T x H x W x 7
            frames = N_cat_aug[:, :, :, :3]
            bg = N_cat_aug[:, :, :, 3:6]
            alphas = N_cat_aug[:, :, :, 6]
            weights = N_cat_aug[:, :, :, 7]
            frames = np.clip(frames, 0, 255)
            bg = np.clip(bg, 0, 255)
            alphas = np.clip(alphas, 0, 255)
        else:
            if self.random.uniform(0, 1) < 0.9:
                # Transform alphas to T x H x W x n_inst
                alphas = alphas.reshape(len(frames), -1, *alphas.shape[1:])
                n_inst = alphas.shape[1]
                alphas = alphas.transpose((0, 2, 3, 1)) # T x H x W x n_inst
                # weights = weights.reshape(len(frames), -1, *weights.shape[1:])
                # weights = weights.transpose((0, 2, 3, 1))
                # N_cat = np.concatenate([frames, alphas, weights], axis=-1) # T x H x W x (n_inst + 3)
                N_cat = np.concatenate([frames, alphas], axis=-1)
                N_cat = N_cat.transpose((1, 2, 3, 0)) # H x W x (n_inst + 3) x T
                N_cat = N_cat.reshape(*N_cat.shape[:2], -1) # H x W x (n_inst + 3)T
                N_cat_aug = self.motion_aug(image=N_cat)["image"] # H x W x (n_inst + 3)T
                N_cat_aug = N_cat_aug.reshape(*N_cat_aug.shape[:2], -1, frames.shape[0]) # H x W x (n_inst + 3) x T
                N_cat_aug = N_cat_aug.transpose((3, 0, 1, 2)) # T x H x W x (n_inst + 3)
                frames = N_cat_aug[:, :, :, :3]
                alphas = N_cat_aug[:, :, :, 3:3 + n_inst]
                # weights = N_cat_aug[:, :, :, 3 + n_inst:]
                frames = np.clip(frames, 0, 255)
                alphas = np.clip(alphas, 0, 255)
                # weights = np.clip(weights, 0, 255)
                alphas = alphas.transpose((0, 3, 1, 2)) # T x n_inst x H x W
                alphas = alphas.reshape(-1, *alphas.shape[2:])
                # weights = weights.transpose((0, 3, 1, 2))
                # weights = weights.reshape(-1, *weights.shape[2:])
            if self.random.uniform(0, 1) < 0.3 and bg is not None:
                N_cat = bg 
                N_cat = N_cat.transpose((1, 2, 3, 0)) # H x W x 3 x T
                N_cat = N_cat.reshape(*N_cat.shape[:2], -1) # H x W x 3T
                N_cat_aug = self.motion_aug(image=N_cat)["image"] # H x W x 3T
                N_cat_aug = N_cat_aug.reshape(*N_cat_aug.shape[:2], -1, frames.shape[0]) # H x W x 3 x T
                N_cat_aug = N_cat_aug.transpose((3, 0, 1, 2)) # T x H x W x 3
                bg = N_cat_aug
                bg = np.clip(bg, 0, 255)
        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        # input_dict["weights"] = weights
        if bg is not None:
            input_dict["bg"] = bg
            input_dict["fg"] = frames
        return input_dict
