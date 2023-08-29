import os
import glob
import numpy as np
import argparse
import logging
import torch
from torch.nn import functional as F
import json
from PIL import Image
from vm2m.utils import CONFIG
from vm2m.network import build_model
from tqdm import tqdm
from vm2m.utils.postprocessing import postprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VM2M demo")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--data-dir", type=str, help="Path to data directory")
    parser.add_argument("--output", type=str, help="output directory")
    parser.add_argument("--skip", type=int, default=1, help="skip frames")
    # parser.add_argument("--temp", action="store_true", help="use temperature memory")
    args = parser.parse_args()

    # Prepare model
    device = f"cuda:0"
    CONFIG.merge_from_file(args.config)
    model = build_model(CONFIG.model)
    model = model.to(device)
    model.eval()
    
    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        logging.warn("Missing keys: {}".format(missing_keys))
        logging.warn("Unexpected keys: {}".format(unexpected_keys))

    # Load frames and coarse annotations
    image_i = 0
    image_dirs = list(os.listdir(os.path.join(args.data_dir, "fgr")))
    for image_dir in tqdm(image_dirs):
        image_i += 1
        if image_i < args.skip:
            continue
        image_paths = sorted(glob.glob(os.path.join(args.data_dir, "fgr", image_dir, "*.jpg")))
        frames = []
        masks = []
        
        # Check no.masks inputs and outputs
        all_out_names = glob.glob(os.path.join(args.output, os.path.basename(image_dir)) + "/*/*.png")
        all_mask_names = glob.glob(os.path.join(args.data_dir, "mask", image_dir) + "/*/*.png")
        if len(all_out_names) == len(all_mask_names):
            continue

        for image_path in image_paths:
            img = Image.open(image_path).convert("RGB")
            img = np.array(img).astype(np.float32) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            frames.append(img)
            inst_masks = []

            mask_dir = os.path.join(args.data_dir, "mask", image_dir, os.path.basename(image_path).replace(".jpg", ""))
            mask_names = sorted(list(os.listdir(mask_dir)))
            for mask_name in mask_names:
                mask_path = os.path.join(mask_dir, mask_name)
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask).astype(np.float32) / 255.
                mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
                inst_masks.append(mask)
            inst_masks = torch.cat(inst_masks, dim=1)
            masks.append(inst_masks)
        # import pdb; pdb.set_trace()
        frames = torch.cat(frames, dim=0)
        instances = torch.cat(masks, dim=0)

        

        # Resize short
        ori_h, ori_w = frames.shape[2:]
        ratio = 768.0 / min(ori_h, ori_w)
        h, w = int(ori_h * ratio), int(ori_w * ratio)
        frames = F.interpolate(frames, size=(h, w), mode="bilinear", align_corners=False)
        instances = F.interpolate(instances, size=(h, w), mode="nearest")

        # Pad to 32
        pad_h = 32 - h % 32
        pad_w = 32 - w % 32
        frames = F.pad(frames, (0, pad_w, 0, pad_h), mode="constant", value=0)
        instances = F.pad(instances, (0, pad_w, 0, pad_h), mode="constant", value=0)
        
        # Normalize
        frames = frames - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        frames = frames / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # Resize instance masks
        # instances = F.max_pool2d(instances, kernel_size=8, stride=8, padding=0)
        instances = (instances > 0.5).float()

        # Start inference
        start_idx = 0
        CLIP_LEN = max(1, 10 // instances.shape[1])
        OVERLAP = 0

        while start_idx < len(frames):
            # print("Processing", start_idx)
            cur_frames = frames[start_idx: start_idx + CLIP_LEN][None]
            cur_masks = instances[start_idx: start_idx + CLIP_LEN][None]
            alphas = None
            if cur_masks.sum() > 0:
                with torch.no_grad():
                    cur_frames = cur_frames.to(device)
                    cur_masks = cur_masks.to(device)
                    
                    alpha_i = 0
                    alphas = []
                    while alpha_i < cur_masks.shape[2]:
                        output = model({"image": cur_frames, "mask": cur_masks[:, :, alpha_i: alpha_i + 10]}, mem_feat=[], mem_query=None, mem_details=None)
                        cur_alphas = output["refined_masks"][0]
                        alpha_i += 10
                        alphas.append(cur_alphas)
                    alphas = torch.cat(alphas, dim=1)

                    # Reversed transform
                    alphas = alphas[:, :, :h, :w]
                    weight = (cur_masks[0].sum((1, 2, 3), keepdim=True) > 0)
                    alphas = alphas * weight
                    alphas = F.interpolate(alphas, size=(ori_h, ori_w), mode="bilinear", align_corners=False)
                    alphas = alphas.cpu().numpy()
            else:
                # Empty mask
                alphas = np.zeros((*cur_masks.shape[1:3], ori_h, ori_w))

            # Save alphas
            for i in range(CLIP_LEN):
                for inst_i in range(alphas.shape[1]):
                    if start_idx > 0 and i < OVERLAP:
                        continue
                    if i >= len(alphas):
                        break
                    
                    img = Image.fromarray((alphas[i, inst_i] * 255).astype(np.uint8))
                    image_name = image_paths[start_idx + i].split("/")[-1].replace(".jpg", "")
                    video_name = image_paths[start_idx + i].split("/")[-2]
                    output_path = os.path.join(args.output, video_name, image_name, f"{inst_i:02d}.png")
                    # import pdb; pdb.set_trace()
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    img.save(output_path)

            start_idx += CLIP_LEN - OVERLAP


