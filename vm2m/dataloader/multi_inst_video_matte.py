import os
import glob
import numpy as np
import cv2
import logging
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from vm2m.utils.utils import resizeAnyShape
try:
    from . import transforms as T
    from .utils import gen_transition_temporal_gt, gen_transition_gt, gen_diff_mask
except ImportError:
    import transforms as T
    from utils import gen_transition_gt, gen_transition_temporal_gt, gen_diff_mask

class MultiInstVidDataset(Dataset):
    def __init__(self, root_dir, split, clip_length, overlap=2, padding_inst=10, is_train=False, short_size=576, 
                    crop=[512, 512], flip_p=0.5, bin_alpha_max_k=30,
                    max_step_size=5, random_seed=2023, modify_mask_p=0.1, mask_dir_name='', downscale_mask_p=0.5, pha_dir='pha', weight_mask_dir='', is_ss_dataset=False, **kwargs):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.is_train = is_train
        self.clip_length = clip_length
        self.overlap = overlap
        self.padding_inst = padding_inst
        self.mask_dir_name = mask_dir_name
        self.pha_dir = pha_dir
        self.weight_mask_dir = weight_mask_dir
        self.is_ss_dataset = is_ss_dataset

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
        if self.is_train or self.mask_dir_name == '':
            self.transforms.append(T.GenMaskFromAlpha(1.0))
        if self.is_train and not is_ss_dataset:
            # self.transforms.append(T.RandomBinarizedMask(self.random, bin_alpha_max_k))
            self.transforms.append(T.ChooseOne(self.random, [
                T.ModifyMaskBoundary(self.random, modify_mask_p),
                T.Compose([
                    T.RandomBinarizedMask(self.random, bin_alpha_max_k),
                    T.DownUpMask(self.random, 0.125, downscale_mask_p)
                ])
            ]))
            # self.transforms.append(T.Compose([
            #         T.RandomBinarizedMask(self.random, bin_alpha_max_k),
            #         T.DownUpMask(self.random, 0.125, downscale_mask_p)
            #     ]))
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
        fg_dir = os.path.join(self.root_dir, self.pha_dir)
        for video_name in sorted(os.listdir(fg_dir)):
            self.load_video_frame(video_name, overlap)
        # self.frame_ids = [x for x in self.frame_ids if x[0] == '6_production_id_4880458_2160p']
        # self.frame_ids = self.frame_ids[-100:]
    
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
            alpha_path = list(glob.glob(os.path.join(self.root_dir, self.pha_dir, video_name, alpha_dir, "*.png")))
            alpha_path = sorted(alpha_path)
            if len(alpha_path) > self.padding_inst:
                alpha_path = alpha_path[:self.padding_inst]
            alpha_paths.extend(alpha_path)

        # In training, drop randomly an instance:
        if self.is_train and self.random.rand() < 0.2 and not self.is_ss_dataset:
            n_inst = len(alpha_paths) // len(frame_paths)
            if n_inst > 1:
                drop_inst_id = self.random.randint(0, n_inst)
                new_alpha_paths = []
                for j in range(len(alpha_paths)):
                    if j % n_inst != drop_inst_id:
                        new_alpha_paths.append(alpha_paths[j])
                alpha_paths = new_alpha_paths

        # import pdb; pdb.set_trace()
        # TODO: Debug the mask guidance
        # if len(alpha_paths) // len(frame_paths) == 3:
        #     alpha_paths = alpha_paths[2::3]

        mask_paths = None
        if self.mask_dir_name != '' and not self.is_train:
            mask_paths = [x.replace(f'/{self.pha_dir}/', '/' + self.mask_dir_name + '/') for x in alpha_paths]
        weight_paths = [''] * len(alpha_paths)
        if self.weight_mask_dir != '' and self.is_train:
            weight_paths = [x.replace(f'/{self.pha_dir}/', '/' + self.weight_mask_dir + '/') for x in alpha_paths]

        input_dict = {
            "frames": frame_paths,
            "alphas": alpha_paths,
            "masks": mask_paths,
            "weights": weight_paths
        }
        # import pdb; pdb.set_trace()
        output_dict = self.transforms(input_dict)
        frames, alphas, masks, transform_info, weights = output_dict["frames"], output_dict["alphas"], output_dict["masks"], output_dict["transform_info"], output_dict["weights"]

        if not self.is_train:
            alphas = output_dict["ori_alphas"]
        
        if (masks.sum() == 0 or alphas.sum() == 0 or (masks.sum((1, 2, 3)) == 0).any()) and self.is_train:
            logging.error("Mask or alpha is zero: {}".format(idx))
            return self.__getitem__(self.random.randint(0, len(self)))
        
        # Padding instances
        add_padding = self.padding_inst - len(alphas)
        if add_padding > 0 and self.is_train:
            new_alpha = torch.zeros(alphas.shape[0], self.padding_inst, *alphas.shape[2:], dtype=alphas.dtype)
            new_mask = torch.zeros(alphas.shape[0], self.padding_inst, *masks.shape[2:], dtype=masks.dtype)
            new_weight = torch.zeros(alphas.shape[0], self.padding_inst, *weights.shape[2:], dtype=weights.dtype)
            chosen_ids = self.random.choice(range(self.padding_inst), alphas.shape[1], replace=False)
            new_alpha[:, chosen_ids] = alphas
            new_mask[:, chosen_ids] = masks
            new_weight[:, chosen_ids] = weights
            masks = new_mask
            alphas = new_alpha
            weights = new_weight
        
        # Transition GT
        transition_gt = None
        if self.is_train:
            # k_size = self.random.choice(range(2, 5))
            # iterations = np.random.randint(5, 15)

            k_size = self.random.choice(range(2, 5))
            iterations = np.random.randint(3, 7)

            diff = (np.abs(alphas[1:].float() - alphas[:-1].float()) > 5).type(torch.uint8) * 255

            # import pdb; pdb.set_trace()

            # transition_gt = gen_transition_gt(alphas.flatten(0, 1)[:, None], masks.flatten(0, 1)[:, None], k_size, iterations)
            # transition_gt = transition_gt.reshape_as(alphas)
            # import pdb; pdb.set_trace()
            # transition_gt = gen_transition_gt(diff.flatten(0, 1)[:, None], None, k_size, iterations)
            transition_gt = gen_diff_mask(diff.flatten(0, 1)[:, None], k_size, iterations)
            transition_gt = transition_gt.reshape_as(diff)
            transition_gt = torch.cat([torch.ones_like(transition_gt[:1]), transition_gt], dim=0)
            transition_gt = transition_gt.sum(1, keepdim=True).expand_as(transition_gt)
            transition_gt = (transition_gt > 0).type(torch.uint8)

        alphas = alphas * 1.0 / 255
        masks = masks * 1.0 / 255
        weights = weights * 1.0 / 255
        
        # Small masks may caused error in attention
        
       
        # if small_masks.sum() == 0:
        #     print(small_masks.sum())
        if self.is_train:
            small_masks = resizeAnyShape(masks, scale_factor=0.125, use_max_pool=True)
            if small_masks.sum() == 0:
                logging.error("Small masks is zero: {}".format(idx))
                return self.__getitem__(self.random.randint(0, len(self)))
        
        out =  {'image': frames,
                'mask': masks.float(),
                'alpha': alphas.float(), 
                'weight': weights.float()}

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
    # dataset = MultiInstVidDataset(root_dir="/mnt/localssd/VHM/syn", split="train", clip_length=3, overlap=2, padding_inst=10, is_train=True, short_size=576, 
    #                 crop=[512, 512], flip_p=0.5, bin_alpha_max_k=30,
    #                 max_step_size=5, random_seed=2023)
    # dataset = MultiInstVidDataset(root_dir="/mnt/localssd/VHM/syn", split="test", clip_length=8, overlap=2, is_train=False, short_size=576, 
    #                 random_seed=2023)
    # dataset = MultiInstVidDataset(root_dir="/mnt/localssd/syn", split="pexels-train", clip_length=8, overlap=2, padding_inst=10, is_train=False, short_size=576, 
    #                 crop=[512, 512], flip_p=0.5, bin_alpha_max_k=30,
    #                 max_step_size=5, random_seed=2023, mask_dir_name='xmem_rename', pha_dir='xmem_rename', weight_mask_dir='', is_ss_dataset=True)
    # from torch.utils import data as torch_data

    dataset = MultiInstVidDataset(root_dir="/mnt/localssd/syn/benchmark", split="real_qual_filtered", clip_length=3, overlap=2, padding_inst=10, is_train=False, short_size=576, 
                    crop=[512, 512], flip_p=0.5, bin_alpha_max_k=30,
                    max_step_size=5, random_seed=2023, mask_dir_name='xmem', pha_dir='xmem', weight_mask_dir='', is_ss_dataset=False)
    # dataloader = torch_data.DataLoader(
    #     dataset, batch_size=1, shuffle=False, pin_memory=True,
    #     sampler=None,
    #     num_workers=16)
    import shutil
    for batch_i, batch in enumerate(dataset):
        # print(batch_i, len(dataloader))
        # continue
        frames, masks, alphas, transition_gt = batch["image"], batch["mask"], batch["alpha"], batch.get("transition", batch.get("trimap"))
        weights = batch["weight"]
        print(frames.shape, masks.shape, alphas.shape, transition_gt.shape)
        shutil.rmtree("debug", ignore_errors=True)
        os.makedirs("debug", exist_ok=True)
        for idx in range(len(frames)):
            frame = frames[idx]
            mask = masks[idx]
            alpha = alphas[idx]
            transition = transition_gt[idx]
            weight = weights[idx]

            frame = frame * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            frame = (frame * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            
            cv2.imwrite("debug/frame_{}.png".format(idx), frame[:, :, ::-1])
            valid_masks = mask.sum((1,2)) > 0
            for inst_i in range(mask.shape[0]):
                if not valid_masks[inst_i]:
                    continue
                cv2.imwrite("debug/mask_{}_{}.png".format(inst_i, idx), mask[inst_i].numpy() * 255)
                cv2.imwrite("debug/alpha_{}_{}.png".format(inst_i, idx), alpha[inst_i].numpy() * 255)
                cv2.imwrite("debug/weight_{}_{}.png".format(inst_i, idx), weight[inst_i].numpy() * 255)
                # cv2.imwrite("debug/transition_{}_{}.png".format(idx, inst_i), transition[inst_i].numpy() * 120)
            cv2.imwrite("debug/transition_{}.png".format(idx), transition[0].numpy() * 255)
        import pdb; pdb.set_trace()