import os
import numpy as np
import argparse
import logging
import torch
from torch.nn import functional as F
import json
from PIL import Image
import pycocotools.mask as coco_mask
from pycocotools.coco import COCO
from vm2m.utils import CONFIG
from vm2m.network import build_model
from vm2m.utils.postprocessing import postprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VM2M demo")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--data-dir", type=str, help="Path to data directory")
    parser.add_argument("--split", type=str, help="Split of data")
    parser.add_argument("--video", type=str, help="video name")
    parser.add_argument("--output", type=str, help="Path to output images")
    parser.add_argument("--json", type=str, help="name of coco json file")
    parser.add_argument("--save-mask", action="store_true", help="Save coarse mask")

    args = parser.parse_args()

    # Prepare model
    device = f"cuda:0"
    CONFIG.merge_from_file(args.config)
    model = build_model(CONFIG.model)
    model = model.to(device)
    model.eval()
    
    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        logging.warn("Missing keys: {}".format(missing_keys))
        logging.warn("Unexpected keys: {}".format(unexpected_keys))

    # Load coarse annotations
    json_path = os.path.join(args.data_dir, args.json)
    anno_data = json.load(open(json_path, "r"))
    video_info = list(filter(lambda x: x["file_names"][0].split("/")[0] == args.video, anno_data["videos"]))[0]
    vid_id = video_info["id"]
    all_filenames = video_info["file_names"]
    annos = list(filter(lambda x: x["video_id"] == vid_id, anno_data["annotations"]))
    instances = []
    for anno in annos:
        masks = []
        for segment in anno["segmentations"]:
            if segment is None: 
                masks.append(np.zeros((video_info["height"], video_info["width"])))
                continue
            # mask
            if type(segment['counts']) == list:
                rle = coco_mask.frPyObjects([segment], video_info['height'], video_info['width'])
            else:
                rle = [segment]
            m = coco_mask.decode(rle)
            masks.append(m[..., 0])
        masks = np.stack(masks, axis=0)
        instances.append(masks)
    instances = np.stack(instances, axis=0)
    instances = torch.from_numpy(instances)
    instances = instances.float()

    if args.save_mask:
        for ins_id in range(instances.shape[0]):
            for idx in range(instances.shape[1]):
                mask = instances[ins_id, idx].numpy()
                mask = Image.fromarray((mask * 255).astype(np.uint8))
                filename = all_filenames[idx].split("/")[-1]
                output_path = os.path.join(args.output, args.split, "mask", args.video, "instance_{}".format(ins_id), filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                mask.save(output_path)

    # Load video
    frames = []
    
    for file_name in all_filenames:
        img = Image.open(os.path.join(args.data_dir, args.split, file_name)).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        frames.append(img)
    
    frames = torch.cat(frames, dim=0)

    # Resize short
    ori_h, ori_w = video_info["height"], video_info["width"]
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
    for ins_id in range(instances.shape[0]):
        start_idx = 0
        CLIP_LEN = 8
        OVERLAP = 2
        while start_idx < len(frames):
            cur_frames = frames[start_idx: start_idx + CLIP_LEN][None]
            cur_masks = instances[ins_id, start_idx: start_idx + CLIP_LEN, None][None]
            alphas = None
            if cur_masks.sum() > 0:
                with torch.no_grad():
                    cur_frames = cur_frames.to(device)
                    cur_masks = cur_masks.to(device)
                    try:
                        output = model({"image": cur_frames, "mask": cur_masks})
                        alphas = output["refined_masks"][0]
                        # Reversed transform
                        alphas = alphas[:, :, :h, :w]
                        alphas = F.interpolate(alphas, size=(ori_h, ori_w), mode="bilinear", align_corners=False)
                        alphas = alphas.squeeze(1).cpu().numpy()
                        # Post-processing
                        # alphas = postprocess(alphas)
                    except:
                        # Save masks only
                        print("Error, save empty mask", start_idx, ins_id)
                        alphas = np.zeros((len(cur_frames), ori_h, ori_w))
            else:
                # Save masks only
                print("Empty mask", start_idx, ins_id)
                alphas = np.zeros((len(cur_frames), ori_h, ori_w))

            # Save alphas
            for i in range(CLIP_LEN):
                if start_idx > 0 and i < OVERLAP:
                    continue
                if i >= len(alphas):
                    break
                img = Image.fromarray((alphas[i] * 255).astype(np.uint8))
                filename = all_filenames[start_idx + i].split("/")[-1]
                output_path = os.path.join(args.output, args.split, "alpha_pred", args.video, "instance_{}".format(ins_id), filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img.save(output_path)

            start_idx += CLIP_LEN - OVERLAP


