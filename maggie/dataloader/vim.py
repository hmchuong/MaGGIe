import os
import glob
import logging

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from maggie.utils.utils import resizeAnyShape

from . import transforms as T
from .utils import gen_transition_gt, gen_diff_mask

class VIMDataset(Dataset):
    def __init__(self, root_dir, split, clip_length, overlap=2, max_step_size=5, max_inst=10, is_train=False, short_size=576, 
                    crop=[512, 512], mask_dir_name='', alpha_dir_name='pha', 
                    padding_crop_p=0.1, flip_p=0.5, gamma_p=0.3, motion_p=0.3, add_noise_p=0.3, jpeg_p=0.1, affine_p=0.1, binarized_kernel=30,
                    random_seed=2023, downscale_mask_p=0.5, **kwargs):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.is_train = is_train
        self.clip_length = clip_length
        self.overlap = overlap
        self.max_inst = max_inst
        self.mask_dir_name = mask_dir_name
        self.alpha_dir_name = alpha_dir_name

        self.video_infos = {} # {video_name: [list of sorted frame names]}
        self.frame_ids = [] # (video_name, start_frame_id)

        if is_train:
            # If train, load all frame index with step 1
            self.load_frame_ids(clip_length - 1)
            self.max_step_size = max_step_size
            
        else:
            # Else, load clip_length with overlapping
            self.load_frame_ids(overlap)
            
        self.random = np.random.RandomState(random_seed)

        self.transforms = [T.Load(), 
                           T.ResizeShort(short_size, transform_alphas=is_train), 
                           T.PaddingMultiplyBy(64, transform_alphas=is_train), 
                           T.Stack()]
        if self.is_train:
            self.transforms.extend([
                    T.RandomCropByAlpha(crop, self.random, padding_prob=padding_crop_p),
                    T.RandomHorizontalFlip(self.random, flip_p),
                    T.GammaContrast(self.random, p=gamma_p),
                    T.MotionBlur(self.random, p=motion_p),
                    T.AdditiveGaussionNoise(self.random, p=add_noise_p),
                    T.JpegCompression(self.random, p=jpeg_p),
                    T.RandomAffine(self.random, p=affine_p)
                ])
                
        if self.is_train or self.mask_dir_name == '':
            self.transforms.append(T.GenMaskFromAlpha(1.0))
        if self.is_train:
            self.transforms.append(T.Compose([
                    T.RandomBinarizedMask(self.random, binarize_max_k=binarized_kernel),
                    T.DownUpMask(self.random, 0.125, downscale_mask_p),
                    T.CutMask(self.random),
                    T.MaskDropout(self.random)
                ]))
        else:
            if self.mask_dir_name == '':
                self.transforms += [T.DownUpMask(self.random, 0.125, 1.0)]
        self.transforms += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        self.transforms = T.Compose(self.transforms)

    def __len__(self):
        return len(self.frame_ids)
    
    def load_video_frame(self, video_name, overlap):
        ''' Load video frame from video_name
        '''
        frame_names = sorted(os.listdir(os.path.join(self.root_dir, 'fgr', video_name)))
        self.video_infos[video_name] = frame_names

        # Load frame ids
        start_idx = 0
        upperbound = len(frame_names) - self.clip_length + 1 if self.is_train else len(frame_names) - overlap
        while start_idx < upperbound:
            self.frame_ids.append((video_name, start_idx))
            start_idx += self.clip_length - overlap

    def load_frame_ids(self, overlap):
        ''' Load frames of all videos
        '''
        fg_dir = os.path.join(self.root_dir, self.alpha_dir_name)
        for video_name in sorted(os.listdir(fg_dir)):
            self.load_video_frame(video_name, overlap)
    
    def __getitem__(self, idx):
        video_name, start_frame_id = self.frame_ids[idx]
        frame_names = self.video_infos[video_name]
        if self.is_train:
            # Load frames

            # Load with step side (based on the start_frame_id)
            end_frame_id = start_frame_id + self.clip_length * self.random.randint(1, self.max_step_size + 1)
            end_frame_id = min(end_frame_id, len(frame_names))
            clip_length = min(end_frame_id - start_frame_id, self.clip_length)
            frame_names = np.random.choice(frame_names[start_frame_id: end_frame_id], clip_length, replace=False)
            frame_names = sorted(frame_names)

            # Reverse the order of frames with probability 0.5
            if self.random.rand() > 0.5:
                frame_names = frame_names[::-1]
        else:
            frame_names = self.video_infos[video_name][start_frame_id:start_frame_id+self.clip_length]
        fgr_dir = "fgr"
        frame_paths = [os.path.join(self.root_dir, fgr_dir, video_name, frame_name) for frame_name in frame_names]
        alpha_paths = []
        for frame_name in frame_names:
            alpha_dir = frame_name.replace(".jpg", "")
            alpha_path = list(glob.glob(os.path.join(self.root_dir, self.alpha_dir_name, video_name, alpha_dir, "*.png")))
            alpha_path = sorted(alpha_path)
            if len(alpha_path) > self.padding_inst:
                alpha_path = alpha_path[:self.padding_inst]
            alpha_paths.extend(alpha_path)

        # In training, drop randomly an instance:
        if self.is_train and self.random.rand() < 0.2:
            n_inst = len(alpha_paths) // len(frame_paths)
            if n_inst > 1:
                drop_inst_id = self.random.randint(0, n_inst)
                new_alpha_paths = []
                for j in range(len(alpha_paths)):
                    if j % n_inst != drop_inst_id:
                        new_alpha_paths.append(alpha_paths[j])
                alpha_paths = new_alpha_paths

        mask_paths = None
        if self.mask_dir_name != '' and not self.is_train:
            mask_paths = [x.replace(f'/{self.alpha_dir_name}/', '/' + self.mask_dir_name + '/') for x in alpha_paths]

        input_dict = {
            "frames": frame_paths,
            "alphas": alpha_paths,
            "masks": mask_paths
        }

        output_dict = self.transforms(input_dict)
        frames, alphas, masks, transform_info = output_dict["frames"], output_dict["alphas"], output_dict["masks"], output_dict["transform_info"]

        if not self.is_train:
            alphas = output_dict["ori_alphas"]
        
        if (masks.sum() == 0 or alphas.sum() == 0 or (masks.sum((1, 2, 3)) == 0).any()) and self.is_train:
            logging.error("Mask or alpha is zero: {}".format(idx))
            return self.__getitem__(self.random.randint(0, len(self)))
        
        # Padding instances
        add_padding = self.padding_inst - len(alphas)
        if add_padding > 0 and self.is_train:
            new_alpha = torch.zeros(alphas.shape[0], self.max_inst, *alphas.shape[2:], dtype=alphas.dtype)
            new_mask = torch.zeros(alphas.shape[0], self.max_inst, *masks.shape[2:], dtype=masks.dtype)
            chosen_ids = self.random.choice(range(self.max_inst), alphas.shape[1], replace=False)
            new_alpha[:, chosen_ids] = alphas
            new_mask[:, chosen_ids] = masks
            masks = new_mask
            alphas = new_alpha
        
        # Transition GT
        transition_gt = None
        if self.is_train:

            k_size = self.random.choice(range(2, 5))
            iterations = np.random.randint(3, 7)

            diff = (np.abs(alphas[1:].float() - alphas[:-1].float()) > 5).type(torch.uint8) * 255

            transition_gt = gen_diff_mask(diff.flatten(0, 1)[:, None], k_size, iterations)
            transition_gt = transition_gt.reshape_as(diff)
            transition_gt = torch.cat([torch.ones_like(transition_gt[:1]), transition_gt], dim=0)
            transition_gt = transition_gt.sum(1, keepdim=True).expand_as(transition_gt)
            transition_gt = (transition_gt > 0).type(torch.uint8)

        alphas = alphas * 1.0 / 255
        masks = masks * 1.0 / 255

        if self.is_train:
            small_masks = resizeAnyShape(masks, scale_factor=0.125, use_max_pool=True)
            if small_masks.sum() == 0:
                logging.error("Small masks is zero: {}".format(idx))
                return self.__getitem__(self.random.randint(0, len(self)))
        
        out =  {'image': frames,
                'mask': masks.float(),
                'alpha': alphas.float()}

        if not self.is_train:
            trans = gen_transition_gt(alphas.flatten(0, 1)[:, None])
            trans = trans.reshape_as(alphas)
            trimap = torch.zeros_like(alphas)
            trimap[alphas > 0.5] = 2.0 # FG
            trimap[trans > 0] = 1.0 # Transition
            out.update({'trimap': trimap, 'image_names': frame_paths, 
                        'transform_info': transform_info, 
                        "skip": 0 if start_frame_id == 0 else self.overlap,
                        "is_first": start_frame_id == 0,
                        "is_last": (start_frame_id + self.clip_length) >= len(self.video_infos[video_name])
                        })
        else:
            out.update({'transition': transition_gt.float()})
        return out