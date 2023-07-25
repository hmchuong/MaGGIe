from .video_matte import SingleInstComposedVidDataset
from .image_matte import ImageMatteDataset
from .him import HIMDataset
from .inst_image_matte import ComposedInstImageMatteDataset

def build_dataset(cfg, is_train=True, random_seed=0):
    if cfg.name in ["VideoMatte240K", "polarized_matting"]:
        if is_train:
            dataset = SingleInstComposedVidDataset(root_dir=cfg.root_dir, split=cfg.split, clip_length=cfg.clip_length, is_train=True, short_size=cfg.short_size,
                                                   bg_dir=cfg.bg_dir, max_step_size=cfg.max_step_size, random_seed=random_seed, flip_p=cfg.flip_prob,
                                                   crop=cfg.crop, blur_p=cfg.blur_prob, blur_kernel_size=cfg.blur_kernel_size, blur_sigma=cfg.blur_sigma)
        else:
            dataset = SingleInstComposedVidDataset(root_dir=cfg.root_dir, split=cfg.split, clip_length=cfg.clip_length, is_train=False, short_size=cfg.short_size, overlap=cfg.clip_overlap, use_thresh_mask=cfg.use_thresh_mask)
    elif cfg.name in ["HHM"]:
        if is_train:
            dataset = ImageMatteDataset(root_dir=cfg.root_dir, split=cfg.split, short_size=cfg.short_size, crop=cfg.crop, flip_p=cfg.flip_prob, is_train=is_train, random_seed=random_seed)
        else:
            dataset = ImageMatteDataset(root_dir=cfg.root_dir, split=cfg.split, short_size=cfg.short_size, is_train=is_train)
    elif cfg.name in ["HIM"]:
        if is_train:
            dataset = ComposedInstImageMatteDataset(root_dir=cfg.root_dir, split=cfg.split, bg_dir=cfg.bg_dir, max_inst=cfg.max_inst,  short_size=cfg.short_size, crop=cfg.crop, random_seed=random_seed)
        else:
            dataset = HIMDataset(root_dir=cfg.root_dir, split=cfg.split, short_size=cfg.short_size)
    else:
        raise NotImplementedError
    return dataset