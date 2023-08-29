import os
import glob
import numpy as np
import cv2
import logging
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
try:
    from . import transforms as T
    from .utils import gen_transition_temporal_gt, gen_transition_gt
except ImportError:
    import transforms as T
    from utils import gen_transition_gt, gen_transition_temporal_gt

class MultiInstVidDataset(Dataset):
    def __init__(self, root_dir, split, clip_length, overlap=2, padding_inst=10, is_train=False, short_size=576, 
                    crop=[512, 512], flip_p=0.5, bin_alpha_max_k=30,
                    max_step_size=5, random_seed=2023, **kwargs):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.is_train = is_train
        self.clip_length = clip_length
        self.overlap = overlap
        self.padding_inst = padding_inst

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
                           T.PaddingMultiplyBy(32, transform_alphas=is_train), 
                           T.Stack()]
        if self.is_train:
            self.transforms.extend([
                T.RandomCropByAlpha(crop, self.random),
                T.RandomHorizontalFlip(self.random, flip_p),
                T.GammaContrast(self.random),
                T.MotionBlur(self.random),
                T.AdditiveGaussionNoise(self.random),
                T.JpegCompression(self.random),
                T.RandomAffine(self.random)
            ])
        self.transforms.append(T.GenMaskFromAlpha(1.0))
        if self.is_train:
            self.transforms.append(T.RandomBinarizedMask(self.random, bin_alpha_max_k))
        self.transforms.append(T.DownUpMask(self.random, 0.125, 1.0 if not self.is_train else 0.7))
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
        frame_names = sorted(os.listdir(os.path.join(self.root_dir, "fgr", video_name)))
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
        fg_dir = os.path.join(self.root_dir, "fgr")
        for video_name in os.listdir(fg_dir):
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
            alpha_path = list(glob.glob(os.path.join(self.root_dir, "pha", video_name, alpha_dir, "*.png")))
            alpha_path = sorted(alpha_path)
            if len(alpha_path) > self.padding_inst:
                alpha_path = alpha_path[:self.padding_inst]
            alpha_paths.extend(alpha_path)

        input_dict = {
            "frames": frame_paths,
            "alphas": alpha_paths,
            "masks": None
        }
        output_dict = self.transforms(input_dict)
        frames, alphas, masks, transform_info = output_dict["frames"], output_dict["alphas"], output_dict["masks"], output_dict["transform_info"]

        if not self.is_train:
            alphas = output_dict["ori_alphas"]
        
        if (masks.sum() == 0 or alphas.sum() == 0) and self.is_train:
            return self.__getitem__(self.random.randint(0, len(self)))
        
        # Padding instances
        add_padding = self.padding_inst - len(alphas)
        if add_padding > 0 and self.is_train:
            new_alpha = torch.zeros(alphas.shape[0], self.padding_inst, *alphas.shape[2:], dtype=alphas.dtype)
            new_mask = torch.zeros(alphas.shape[0], self.padding_inst, *masks.shape[2:], dtype=masks.dtype)
            chosen_ids = self.random.choice(range(self.padding_inst), alphas.shape[1], replace=False)
            new_alpha[:, chosen_ids] = alphas
            new_mask[:, chosen_ids] = masks
            masks = new_mask
            alphas = new_alpha
        
        # Transition GT
        transition_gt = None
        if self.is_train:
            k_size = self.random.choice(range(2, 5))
            iterations = np.random.randint(5, 15)
            transition_gt = gen_transition_gt(alphas.flatten(0, 1)[:, None], masks.flatten(0, 1)[:, None], k_size, iterations)
            transition_gt = transition_gt.reshape_as(alphas)

        alphas = alphas * 1.0 / 255
        masks = masks * 1.0 / 255

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
                        "skip": 0 if start_frame_id == 0 else self.overlap})
        else:
            out.update({'transition': transition_gt.float()})
        return out

if __name__ == "__main__":
    # dataset = MultiInstVidDataset(root_dir="/mnt/localssd/VideoMatte240K_syn", split="train", clip_length=8, overlap=2, padding_inst=10, is_train=True, short_size=576, 
    #                 crop=[512, 512], flip_p=0.5, bin_alpha_max_k=30,
    #                 max_step_size=5, random_seed=2023)
    dataset = MultiInstVidDataset(root_dir="/mnt/localssd/VideoMatte240K_syn", split="valid", clip_length=8, overlap=2, is_train=False, short_size=576, 
                    random_seed=2023)
    for batch in dataset:
        frames, masks, alphas, transition_gt = batch["image"], batch["mask"], batch["alpha"], batch.get("transition", batch.get("trimap"))
        print(frames.shape, masks.shape, alphas.shape, transition_gt.shape)
        
        for idx in range(len(frames)):
            frame = frames[idx]
            mask = masks[idx]
            alpha = alphas[idx]
            transition = transition_gt[idx]

            frame = frame * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            frame = (frame * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            cv2.imwrite("debug/frame_{}.png".format(idx), frame[:, :, ::-1])
            for inst_i in range(mask.shape[0]):
                cv2.imwrite("debug/mask_{}_{}.png".format(idx, inst_i), mask[inst_i].numpy() * 255)
                cv2.imwrite("debug/alpha_{}_{}.png".format(idx, inst_i), alpha[inst_i].numpy() * 255)
                cv2.imwrite("debug/transition_{}_{}.png".format(idx, inst_i), transition[inst_i].numpy() * 120)
        import pdb; pdb.set_trace()