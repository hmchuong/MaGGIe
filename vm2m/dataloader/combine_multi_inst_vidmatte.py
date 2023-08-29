import os
from torch.utils.data import Dataset
from .multi_inst_video_matte import MultiInstVidDataset

class CombineMultiInstVideoMatte(Dataset):
    def __init__(self, root_dir, split, clip_length, overlap=2, padding_inst=10, is_train=False, short_size=576, 
                    crop=[512, 512], flip_p=0.5, bin_alpha_max_k=30,
                    max_step_size=5, random_seed=2023, ratio=[2, 1, 1], **kwargs):
        super().__init__()
        self.ratio = ratio
        self.vm240k_syn = MultiInstVidDataset(os.path.join(root_dir, "VideoMatte240K_syn"), split, clip_length, overlap, padding_inst, is_train, short_size, 
                    crop, flip_p, bin_alpha_max_k,
                    max_step_size, random_seed, **kwargs)
        self.ytvis = MultiInstVidDataset(os.path.join(root_dir, "VIS"), "YTVIS", clip_length, overlap, padding_inst, is_train, short_size,
                                                crop, flip_p, bin_alpha_max_k,
                                                max_step_size, random_seed, **kwargs)
        self.ovis = MultiInstVidDataset(os.path.join(root_dir, "VIS"), "OVIS", clip_length, overlap, padding_inst, is_train, short_size,
                                                crop, flip_p, bin_alpha_max_k,
                                                max_step_size, random_seed, **kwargs)
    
    def __len__(self):
        num_data = len(self.vm240k_syn) * self.ratio[0]
        num_data += len(self.ytvis) * self.ratio[1]
        num_data += len(self.ovis) * self.ratio[2]
        return num_data

    def __getitem__(self, idx):
        data_i = idx % sum(self.ratio)
        if data_i < self.ratio[0]:
            return self.vm240k_syn[idx % len(self.vm240k_syn)]
        elif data_i < self.ratio[0] + self.ratio[1]:
            return self.ytvis[idx % len(self.ytvis)]
        else:
            return self.ovis[idx % len(self.ovis)]
