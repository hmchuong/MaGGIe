from .him import HIMDataset
from .vim import VIMDataset

def build_dataset(cfg, is_train=True, random_seed=0):
    if cfg.name in ["HIM"]:
        if is_train:
            dataset = HIMDataset(root_dir=cfg.root_dir, split=cfg.split, max_inst=cfg.max_inst, short_size=cfg.short_size, 
                                 crop=cfg.crop, is_train=is_train, random_seed=random_seed, alpha_dir_name=cfg.alpha_dir_name, mask_dir_name=cfg.mask_dir_name,
                                 padding_crop_p=cfg.padding_crop_p, flip_p=cfg.flip_p, gamma_p=cfg.gamma_p, add_noise_p=cfg.add_noise_p, jpeg_p=cfg.jpeg_p, affine_p=cfg.affine_p, 
                                 binarized_kernel=cfg.binarized_kernel, downscale_mask_p=cfg.downscale_mask_p)
        else:
            dataset = HIMDataset(root_dir=cfg.root_dir, split=cfg.split, short_size=cfg.short_size, is_train=is_train, 
                                 downscale_mask_p=0 if cfg.downscale_mask else 1, alpha_dir_name=cfg.alpha_dir_name, mask_dir_name=cfg.mask_dir_name)
    elif cfg.name in ["VIM"]:
        if is_train:
            dataset = VIMDataset(root_dir=cfg.root_dir, split=cfg.split, is_train=is_train, alpha_dir_name=cfg.alpha_dir_name, mask_dir_name=cfg.mask_dir_name,
                                 clip_length=cfg.clip_length, max_step_size=cfg.max_step_size, max_inst=cfg.max_inst, short_size=cfg.short_size, crop=cfg.crop, 
                                 padding_crop_p=cfg.padding_crop_p, flip_p=cfg.flip_p, gamma_p=cfg.gamma_p, motion_p=cfg.motion_p, add_noise_p=cfg.add_noise_p, 
                                 jpeg_p=cfg.jpeg_p, affine_p=cfg.affine_p, binarized_kernel=cfg.binarized_kernel, downscale_mask_p=cfg.downscale_mask_p, random_seed=random_seed)
        else:
            dataset = VIMDataset(root_dir=cfg.root_dir, split=cfg.split, clip_length=cfg.clip_length, overlap=cfg.clip_overlap, is_train=is_train, 
                                 short_size=cfg.short_size, mask_dir_name=cfg.mask_dir_name, alpha_dir_name=cfg.alpha_dir_name)
    else:
        raise NotImplementedError
    return dataset