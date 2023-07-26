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
from vm2m.utils.postprocessing import postprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VM2M demo")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--image-dir", type=str, help="Path to data directory")
    parser.add_argument("--mask-dir", type=str, help="Path to output images")
    parser.add_argument("--output", type=str, help="name of coco json file")

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
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    frames = []
    masks = []
    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        frames.append(img)
        mask_path = os.path.join(args.mask_dir, os.path.basename(image_path).replace(".jpg", ".png"))
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask).astype(np.float32) / 255.
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        else:
            mask = torch.zeros((1, 1, img.shape[2], img.shape[3]))
        masks.append(mask)

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
    instances = F.max_pool2d(instances, kernel_size=8, stride=8, padding=0)
    
    # Start inference
    start_idx = 0
    CLIP_LEN = 8
    OVERLAP = 2
    while start_idx < len(frames):
        cur_frames = frames[start_idx: start_idx + CLIP_LEN][None]
        cur_masks = instances[start_idx: start_idx + CLIP_LEN, None][None]
        alphas = None
        if cur_masks.sum() > 0:
            with torch.no_grad():
                cur_frames = cur_frames.to(device)
                cur_masks = cur_masks.to(device)
                # try:
                # import pdb; pdb.set_trace()
                output = model({"image": cur_frames, "mask": cur_masks})
                alphas = output["refined_masks"][0]
                # Reversed transform
                alphas = alphas[:, :, :h, :w]
                alphas = F.interpolate(alphas, size=(ori_h, ori_w), mode="bilinear", align_corners=False)
                alphas = alphas.squeeze(1).cpu().numpy()
                # except:
                #     # Save masks only
                #     print("Error, save empty mask", start_idx, ins_id)
                #     alphas = np.zeros((len(cur_frames), ori_h, ori_w))
        else:
            # Save masks only
            print("Empty mask", start_idx)
            alphas = np.zeros((len(cur_frames), ori_h, ori_w))

        # Save alphas
        for i in range(CLIP_LEN):
            if start_idx > 0 and i < OVERLAP:
                continue
            if i >= len(alphas):
                break
            img = Image.fromarray((alphas[i] * 255).astype(np.uint8))
            filename = image_paths[start_idx + i].split("/")[-1].replace(".jpg", ".png")
            output_path = os.path.join(args.output, filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)

        start_idx += CLIP_LEN - OVERLAP


