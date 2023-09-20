from .video_matte import SingleInstComposedVidDataset
from .image_matte import ImageMatteDataset
from .him import HIMDataset
from .inst_image_matte import ComposedInstImageMatteDataset
from .inst_video_matte import ComposedInstVidDataset
from .multi_inst_video_matte import MultiInstVidDataset
from .combine_multi_inst_vidmatte import CombineMultiInstVideoMatte

def build_dataset(cfg, is_train=True, random_seed=0):
    if cfg.name in ["VideoMatte240K", "polarized_matting"]:
        if is_train:
            dataset = SingleInstComposedVidDataset(root_dir=cfg.root_dir, split=cfg.split, clip_length=cfg.clip_length, is_train=True, short_size=cfg.short_size,
                                                   bg_dir=cfg.bg_dir, max_step_size=cfg.max_step_size, random_seed=random_seed, flip_p=cfg.flip_prob,
                                                   crop=cfg.crop, blur_p=cfg.blur_prob, blur_kernel_size=cfg.blur_kernel_size, blur_sigma=cfg.blur_sigma)
        else:
            dataset = SingleInstComposedVidDataset(root_dir=cfg.root_dir, split=cfg.split, clip_length=cfg.clip_length, is_train=False, short_size=cfg.short_size, overlap=cfg.clip_overlap, use_thresh_mask=cfg.use_thresh_mask)
    elif cfg.name in ["InstVM240K"] and is_train:
        dataset = ComposedInstVidDataset(max_inst=cfg.max_inst, padding_inst=cfg.padding_inst, root_dir=cfg.root_dir, split=cfg.split, clip_length=cfg.clip_length, is_train=True, short_size=cfg.short_size,
                                                   bg_dir=cfg.bg_dir, max_step_size=cfg.max_step_size, random_seed=random_seed, flip_p=cfg.flip_prob,
                                                   crop=cfg.crop, blur_p=cfg.blur_prob, blur_kernel_size=cfg.blur_kernel_size, blur_sigma=cfg.blur_sigma)
    elif cfg.name in ["HHM"]:
        if is_train:
            dataset = ImageMatteDataset(root_dir=cfg.root_dir, split=cfg.split, short_size=cfg.short_size, crop=cfg.crop, flip_p=cfg.flip_prob, is_train=is_train, random_seed=random_seed)
        else:
            dataset = ImageMatteDataset(root_dir=cfg.root_dir, split=cfg.split, short_size=cfg.short_size, is_train=is_train)
    elif cfg.name in ["HIM"]:
        if is_train:
            # dataset = ComposedInstImageMatteDataset(root_dir=cfg.root_dir, split=cfg.split, bg_dir=cfg.bg_dir, max_inst=cfg.max_inst, padding_inst=cfg.padding_inst, short_size=cfg.short_size, crop=cfg.crop, random_seed=random_seed, use_single_instance_only=cfg.use_single_instance_only)
            dataset = HIMDataset(root_dir=cfg.root_dir, split=cfg.split, max_inst=cfg.max_inst, padding_inst=cfg.padding_inst, short_size=cfg.short_size, crop=cfg.crop, random_seed=random_seed, is_train=is_train, 
                                flip_p=cfg.flip_prob, downscale_mask=cfg.downscale_mask, mask_dir_name=cfg.mask_dir_name, modify_mask_p=cfg.modify_mask_p, downscale_mask_p=cfg.downscale_mask_p, use_maskrcnn_p=cfg.use_maskrcnn_p)
        else:
            dataset = HIMDataset(root_dir=cfg.root_dir, split=cfg.split, short_size=cfg.short_size, is_train=is_train, downscale_mask=cfg.downscale_mask, alpha_dir_name=cfg.alpha_dir_name, mask_dir_name=cfg.mask_dir_name)
    elif cfg.name in ["MultiInstVideo"]:
        if is_train:
            dataset = MultiInstVidDataset(root_dir=cfg.root_dir, split=cfg.split, 
                                            clip_length=cfg.clip_length, padding_inst=cfg.padding_inst, is_train=True, short_size=cfg.short_size, 
                                            crop=cfg.crop, flip_p=cfg.flip_prob,
                                            max_step_size=cfg.max_step_size, random_seed=random_seed)
        else:
            dataset = MultiInstVidDataset(root_dir=cfg.root_dir, split=cfg.split, clip_length=cfg.clip_length, overlap=cfg.clip_overlap, is_train=False, short_size=cfg.short_size, random_seed=random_seed, mask_dir_name=cfg.mask_dir_name)
    elif cfg.name in ["CombineMultiInstVideo"]:
         if is_train:
            dataset = CombineMultiInstVideoMatte(root_dir=cfg.root_dir, split=cfg.split, 
                                            clip_length=cfg.clip_length, padding_inst=cfg.padding_inst, is_train=True, short_size=cfg.short_size, 
                                            crop=cfg.crop, flip_p=cfg.flip_prob,
                                            max_step_size=cfg.max_step_size, random_seed=random_seed, pha_dir=cfg.pha_dir, weight_mask_dir=cfg.weight_mask_dir)
         else:
             raise NotImplementedError("CombineMultiInstVideo is only for training")
    else:
        raise NotImplementedError
    return dataset