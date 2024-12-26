import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
from PIL import Image
sys.path.append("./samurai/sam2")
from sam2.build_sam import build_sam2_video_predictor

from maskrcnn import predict_human_mask

color = [(255, 0, 0)]

MAX_FRAMES = 100
MAX_FPS = 12
DEFAULT_FRAME_PATH = "video_results/fgr/video0/"

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def main(args):
    video = args.video_path

    # Extract video to frames
    if os.path.exists("video_results"):
        os.system("rm -rf video_results")
    os.system("mkdir -p video_results")
    os.system("mkdir -p video_results/fgr")
    os.system(f"mkdir -p {DEFAULT_FRAME_PATH}")

    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    fps = min(MAX_FPS, fps)

    os.system(f"ffmpeg -i {video} -vf fps={fps} {DEFAULT_FRAME_PATH}/%04d.jpg")

    frame_names = sorted(os.listdir(DEFAULT_FRAME_PATH))
    if len(frame_names) > MAX_FRAMES:
        print(f"Too many frames ({len(frame_names)}), only process the first {MAX_FRAMES} frames.")
        for frame_name in frame_names[MAX_FRAMES:]:
            frame_path = os.path.join(DEFAULT_FRAME_PATH, frame_name)
            os.remove(frame_path)
        frame_names = frame_names[:MAX_FRAMES]
        
    # Find human masks in the first 10 frame using Mask-RCNN
    init_masks = None
    mask_index = 0

    for i in range(min(10, len(frame_names)// fps)):
        frame_path = os.path.join(DEFAULT_FRAME_PATH, frame_names[i * fps])
        image = Image.open(frame_path)
        _, init_masks = predict_human_mask(image)

        # Save masks as png images
        if init_masks.max() > 0:
            mask_index = i
            break
    
    if init_masks.max() == 0:
        return
        
    for frame_name in frame_names[:mask_index]:
        frame_path = os.path.join(DEFAULT_FRAME_PATH, frame_name)
        os.remove(frame_path)
    
    frame_names = frame_names[mask_index:]
    

    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    loaded_frames = [cv2.imread(os.path.join(DEFAULT_FRAME_PATH, frame_name)) for frame_name in frame_names]
    height, width = loaded_frames[0].shape[:2]

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(DEFAULT_FRAME_PATH, offload_video_to_cpu=True)
        
        for instance_id in range(1, init_masks.max() + 1):
            ys, xs = np.where(init_masks == instance_id)
            bbox = (xs.min(), ys.min(), xs.max(), ys.max())
            bbox = [int(x) for x in bbox]
            # print(bbox)
            

            _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=instance_id - 1)
            # break
            

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                mask = mask.astype("uint8") * 255
                out_path = f"video_results/mask/video0/" + frame_names[frame_idx].replace(".jpg", "") +  f"/{obj_id:02d}.png"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, mask)
            

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--model_path", default="samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    args = parser.parse_args()
    main(args)
