import os
import glob
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import transforms as T
from .utils import gen_transition_gt

class HIMDataset(Dataset):
    def __init__(self, root_dir, split, max_inst=10, short_size=768, is_train=False, random_seed=2023, \
                crop=(512, 512), padding_crop_p=0.1, flip_p=0.5, gamma_p=0.3, add_noise_p=0.3, jpeg_p=0.1, affine_p=0.1, 
                binarized_kernel=30, downscale_mask_p=0.5, alpha_dir_name='alphas', mask_dir_name='', \
                **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.alpha_dir_name = alpha_dir_name
        self.mask_dir_name = mask_dir_name
        self.is_train = is_train

        self.short_size = short_size
        self.max_inst = max_inst
        self.downscale_mask = downscale_mask_p > 0
        
        self.random = np.random.RandomState(random_seed)

        if not is_train:
            self.prepare_image_train()
        else:
            self.prepare_image_test()

        self.transforms = [
            T.Load(),
            T.ResizeShort(short_size, transform_alphas=is_train),
            T.PaddingMultiplyBy(64, transform_alphas=is_train),
            T.Stack()
        ]
        if self.is_train:
            self.transforms += [
                T.RandomCropByAlpha(crop, self.random, padding_prob=padding_crop_p),
                T.RandomHorizontalFlip(self.random, flip_p),
                T.GammaContrast(self.random, p=gamma_p),
                T.AdditiveGaussionNoise(self.random, p=add_noise_p),
                T.JpegCompression(self.random, p=jpeg_p), 
                T.RandomAffine(self.random, p=affine_p), 
                T.Compose([
                    T.RandomBinarizedMask(self.random, binarized_kernel),
                    T.DownUpMask(self.random, 0.125, downscale_mask_p),
                    T.CutMask(self.random)
                ])
            ]
        else:
            # Downscale alpha as guidance mask when no mask dir is provided
            if self.mask_dir_name == '':
                self.transforms += [T.GenMaskFromAlpha(), T.DownUpMask(self.random, 0.125, 1.0)]
        self.transforms += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        self.transforms = T.Compose(self.transforms)
    
    def prepare_image_train(self):
        images = sorted(glob.glob(os.path.join(self.root_dir, "images", self.split, "*.jpg")))
        all_alphas = []
        target_dir_name = self.alpha_dir_name if self.mask_dir_name == '' else self.mask_dir_name
        valid_images = []
        for image in images:
            image_name = os.path.basename(image).replace(".jpg", "")
            alpha_dir = os.path.join(self.root_dir, target_dir_name, self.split, image_name)
            if not os.path.exists(alpha_dir):
                continue
            valid_images.append(image)
            alphas = sorted(os.listdir(alpha_dir))
            all_alphas.append([os.path.join(self.root_dir, self.alpha_dir_name, self.split, image_name, p) for p in alphas])
        self.data = list(zip(valid_images, all_alphas))
    
    def prepare_image_test(self):
        images = sorted(glob.glob(os.path.join(self.root_dir, self.split, "images", "*.jpg")))
        all_alphas = []
        for image in images:
            image_name = os.path.basename(image).replace(".jpg", "")
            alphas = sorted(glob.glob(os.path.join(self.root_dir, self.split, self.alpha_dir_name, image_name, "*.png")))
            all_alphas.append(alphas)
        self.data = list(zip(images, all_alphas))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path, alphas = self.data[index]

        if len(alphas) > self.max_inst:
            alphas = self.random.choice(alphas, self.max_inst, replace=False)

        # Load mask path and random replace the mask by alpha
        masks = None
        if self.is_train:
            masks = alphas
        elif self.mask_dir_name != '':
            masks = [alpha.replace(self.alpha_dir_name, self.mask_dir_name) for alpha in alphas]

        # Load image
        input_dict = {
            "frames": [image_path],
            "alphas": alphas,
            "masks": masks,
            "weights": None
        }
        output_dict = self.transforms(input_dict)
        
        image, alpha, mask, transform_info = output_dict["frames"], output_dict["alphas"], output_dict["masks"], output_dict["transform_info"]
        fg, bg = output_dict["fg"], output_dict["bg"]

        # Remove invalid alpha (< 5% area)
        if self.is_train:
            valid_ids = (alpha > 127).sum((-1, -2)) > (0.001 * alpha.shape[-1] * alpha.shape[-2])
            valid_ids = torch.nonzero(valid_ids)
            alpha = alpha[valid_ids[:, 0], valid_ids[:, 1]]
            mask = mask[valid_ids[:, 0], valid_ids[:, 1]]
            fg = fg[valid_ids[:, 0], valid_ids[:, 1]]
            bg = bg[valid_ids[:, 0], valid_ids[:, 1]]
            if len(alpha.shape) == 3:
                alpha = alpha.unsqueeze(0)
                mask = mask.unsqueeze(0)
                fg = fg.unsqueeze(0)
                bg = bg.unsqueeze(0)

            if mask.numel() == 0:
                logging.warning("Mask is empty after removing tiny masks")
                return self.__getitem__(self.random.randint(0, len(self.data)))

        # remove one random alpha
        if alpha.shape[1] > 1 and self.is_train and self.random.rand() < 0.05:
            num_alphas = alpha.shape[1] - 1
            chosen_ids = self.random.choice(range(alpha.shape[1]), num_alphas, replace=False)
            alpha = alpha[:, chosen_ids]
            mask = mask[:, chosen_ids]
            fg = fg[:, chosen_ids]
            bg = bg[:, chosen_ids]
            if len(alpha.shape) == 3:
                alpha = alpha.unsqueeze(0)
                mask = mask.unsqueeze(0)
                fg = fg.unsqueeze(0)
                bg = bg.unsqueeze(0)
        
        if not self.is_train:
            alpha = output_dict["ori_alphas"]
        if mask.sum() == 0 and self.is_train:
            logging.warning("Mask is empty")
            return self.__getitem__(self.random.randint(0, len(self.data)))

        alpha = alpha * 1.0 / 255
        mask = mask * 1.0 / 255
        add_padding = self.max_inst - alpha.shape[1]
        if add_padding > 0 and self.is_train:
            new_alpha = torch.zeros(1, self.max_inst, *alpha.shape[2:])
            new_mask = torch.zeros(1, self.max_inst, *mask.shape[2:])
            chosen_ids = self.random.choice(range(self.max_inst), alpha.shape[1], replace=False)
            new_alpha[:, chosen_ids] = alpha
            new_mask[:, chosen_ids] = mask
            mask = new_mask
            alpha = new_alpha

            new_fg = torch.zeros(1, self.max_inst, *fg.shape[2:])
            new_bg = torch.zeros(1, self.max_inst, *bg.shape[2:])
            new_fg[:, chosen_ids] = fg
            new_bg[:, chosen_ids] = bg
            fg = new_fg
            bg = new_bg
        if self.downscale_mask:
            mask = F.interpolate(mask, size=(image.shape[2] // 8, image.shape[3] // 8), mode="nearest")

        out =  {
            'image': image, 
            'mask': mask.float(),
            'alpha': alpha.float(),
            'fg': fg.float(),
            'bg': bg.float()
        }
        
        # Generate trimap for evaluation
        if self.is_train:
            k_size = self.random.choice(range(2, 5))
            iterations = self.random.randint(5, 15)
            trans = gen_transition_gt(alpha[0, :, None], mask[0, :, None], k_size=k_size, iterations=iterations)
            out.update({'transition': trans.float()[None, :, 0]})
        else:
            trans = gen_transition_gt(alpha[0, :, None])
            trans = trans.squeeze(1)[None]
            trimap = torch.zeros_like(alpha)
            trimap[alpha > 0.5] = 2.0 # FG
            trimap[trans > 0] = 1.0 # Transition
            out.update({'trimap': trimap, 
                        'image_names': [image_path], 
                        'alpha_names': [os.path.basename(a) for a in alphas], 
                        'transform_info': transform_info, 
                        "skip": 0})
            
        return out
