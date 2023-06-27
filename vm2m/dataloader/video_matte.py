import os
import numpy as np
import cv2
import logging
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
try:
    from . import transforms as T
    from .utils import gen_transition_gt
except:
    import transforms as T
    from utils import gen_transition_gt

class SingleInstComposedVidDataset(Dataset):
    def __init__(self, root_dir, split, clip_length, overlap=0, short_size=1024, 
                 is_train=False, 
                 crop=[512, 512], flip_p=0.5, bin_alpha_max_k=30,
                 blur_p=0.5, blur_kernel_size=[5, 15, 25], blur_sigma=[1.0, 1.5, 3.0, 5.0],
                 bg_dir=None, max_step_size=5, random_seed=2023, **kwargs):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.is_train = is_train
        self.clip_length = clip_length
        self.overlap = overlap

        self.video_infos = {} # {video_name: [list of sorted frame names]}
        self.frame_ids = [] # (video_name, start_frame_id)


        if is_train:
            # If train, load all frame index with step 1
            self.load_frame_ids(clip_length - 1)
            self.max_step_size = max_step_size
            self.random = np.random.RandomState(random_seed)
        else:
            # Else, load clip_length with overlapping
            self.load_frame_ids(overlap)
        
        self.transforms = [T.Load(), T.ResizeShort(short_size, transform_alphas=is_train), T.PaddingMultiplyBy(32, transform_alphas=is_train), T.Stack()]
        if self.is_train:
            bg_images = self.load_bg(bg_dir)
            self.transforms.extend([
                T.RandomCropByAlpha(crop, self.random),
                T.RandomHorizontalFlip(self.random, flip_p),
                T.LoadRandomBackground(bg_images, self.random, 
                                          blur_p=blur_p, 
                                          blur_kernel_size=blur_kernel_size, 
                                          blur_sigma=blur_sigma),
                T.GammaContrast(self.random),
                T.HistogramMatching(self.random),
                T.MotionBlur(self.random),
                T.AdditiveGaussionNoise(self.random),
                T.JpegCompression(self.random),
                T.RandomAffine(self.random),
                T.ComposeBackground(),
                T.RandomBinarizeAlpha(self.random, bin_alpha_max_k),
            ])
        self.transforms += [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
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
        upperbound = len(frame_names) - self.clip_length + 1 if self.is_train else len(frame_names) - overlap + 1
        while start_idx < upperbound:
            self.frame_ids.append((video_name, start_idx))
            start_idx += self.clip_length - overlap

    def load_frame_ids(self, overlap):
        ''' Load frames of all videos
        '''
        fg_dir = os.path.join(self.root_dir, "fgr")
        if not self.is_train:
            fg_dir = os.path.join(self.root_dir, "comp")
        for video_name in os.listdir(fg_dir):
            self.load_video_frame(video_name, overlap)
    
    def load_bg(self, bg_dir):
        ''' Load background image paths
        '''
        bg_images = []
        for image_name in os.listdir(bg_dir):
            image_path = os.path.join(bg_dir, image_name)
            bg_images.append(image_path)
        return bg_images
    
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

        fgr_dir = "fgr" if self.is_train else "comp"
        frame_paths = [os.path.join(self.root_dir, fgr_dir, video_name, frame_name) for frame_name in frame_names]
        mask_paths = None
        if not self.is_train:
            mask_paths = [os.path.join(self.root_dir, "coarse", video_name, frame_name) for frame_name in frame_names]
        alpha_paths = [os.path.join(self.root_dir, "pha", video_name, frame_name) for frame_name in frame_names]
        
        # Transforms
        input_dict = {
            "frames": frame_paths,
            "alphas": alpha_paths,
            "masks": mask_paths
        }
        output_dict = self.transforms(input_dict)
        frames, alphas, masks, transform_info = output_dict["frames"], output_dict["alphas"], output_dict["masks"], output_dict["transform_info"]

        masks = F.interpolate(masks, size=(masks.shape[2] // 8, masks.shape[3] // 8), mode="nearest")

        # Transition GT
        transition_gt = None
        if self.is_train:
            k_size = self.random.choice(range(2, 5))
            iterations = np.random.randint(5, 15)
            transition_gt = gen_transition_gt(alphas, masks, k_size, iterations)

        alphas = alphas * 1.0 / 255
        masks = masks * 1.0 / 255

        if masks.sum() == 0:
            logging.error("Get another sample, alphas are incorrect: {} - {}".format(alpha_paths[0], alpha_paths[-1]))
            return self.__getitem__(self.random.randint(0, len(self.frame_ids)))
        
        out =  {'image':frames, 'mask': masks.float(), 'alpha': alphas.float()}
        out['fg'] = output_dict.get('fg', frames)
        out['bg'] = output_dict.get('bg', frames)
        
        if "ignore_regions" in output_dict:
            ignore_regions = output_dict["ignore_regions"] < 0.5
            ignore_regions = torch.from_numpy(ignore_regions)[:, None]
            transition_gt[ignore_regions] = 0
            # mask out transition GT
            
        if not self.is_train:
            # Generate trimap for evaluation
            trans = gen_transition_gt(alphas)
            trimap = torch.zeros_like(alphas)
            trimap[alphas > 0.5] = 2.0 # FG
            trimap[trans > 0] = 1.0 # Transition
            out.update({'trimap': trimap, 'image_names': frame_paths, 'transform_info': transform_info, "skip": 0 if start_frame_id == 0 else self.overlap})
        else:
            out.update({'transition': transition_gt.float()})
        return out

if __name__ == "__main__":
    train_dataset = SingleInstComposedVidDataset(root_dir="/home/chuongh/mask2matte/data/VideoMatte240K", split="train", clip_length=8, bg_dir="/mnt/localssd/bg", max_step_size=5, is_train=True)
    valid_dataset = SingleInstComposedVidDataset(root_dir="/home/chuongh/mask2matte/data/VideoMatte240K", split="valid", clip_length=8, is_train=False)
    
    for frames, masks, alphas, transition_gt in train_dataset:
        for idx in range(len(frames)):
            frame = frames[idx]
            mask = masks[idx]
            alpha = alphas[idx]
            transition = transition_gt[idx]

            frame = frame * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            frame = (frame * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            cv2.imwrite("frame_{}.png".format(idx), frame[:, :, ::-1])
            cv2.imwrite("mask_{}.png".format(idx), mask[0].numpy() * 255)
            cv2.imwrite("alpha_{}.png".format(idx), alpha[0].numpy() * 255)
            cv2.imwrite("transition_{}.png".format(idx), transition[0].numpy() * 255)
        import pdb; pdb.set_trace()