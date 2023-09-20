import os
from torch.utils.data import Dataset
from .multi_inst_video_matte import MultiInstVidDataset

class CombineMultiInstVideoMatte(Dataset):
    def __init__(self, root_dir, split, clip_length, overlap=2, padding_inst=10, is_train=False, short_size=576, 
                    crop=[512, 512], flip_p=0.5, bin_alpha_max_k=30,
                    max_step_size=5, random_seed=2023, ratio=[2, 1], pha_dir='pha', weight_mask_dir='', **kwargs):
        super().__init__()
        self.ratio = ratio

        # Syn dataset
        self.vm240k_syn = MultiInstVidDataset(os.path.join(root_dir, "syn"), split, clip_length, overlap, padding_inst, is_train, short_size, 
                    crop, flip_p, bin_alpha_max_k,
                    max_step_size, random_seed, pha_dir='pha', **kwargs)
        
        # Pseudo-labels
        self.vipseg = MultiInstVidDataset(os.path.join(root_dir, "VIPSeg"), "out", clip_length, overlap, padding_inst, is_train, short_size,
                                                crop, flip_p, bin_alpha_max_k,
                                                max_step_size, random_seed, pha_dir=pha_dir, weight_mask_dir=weight_mask_dir, **kwargs)
    
    def __len__(self):
        num_data = len(self.vm240k_syn) * self.ratio[0]
        num_data += len(self.vipseg) * self.ratio[1]
        return num_data

    def __getitem__(self, idx):
        data_i = idx % sum(self.ratio)
        if data_i < self.ratio[0]:
            return self.vm240k_syn[idx % len(self.vm240k_syn)]
        else:
            return self.vipseg[idx % len(self.vipseg)]
