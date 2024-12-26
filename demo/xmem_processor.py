import os
import sys
sys.path.append("./XMem")

import cv2
import torch
from PIL import Image
from model.network import XMem
from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask
from maskrcnn import predict_human_mask

MAX_FRAMES = 100
MAX_FPS = 12
MAX_SIZE = 640
DEFAULT_FRAME_PATH = "video_results/fgr/video0/"

if os.path.exists("XMem.pth") == False:
    os.system("wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth")

config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}

network = XMem(config, 'XMem.pth').eval().cuda()

def process_video(video):
    # Extract video to frames
    if os.path.exists("video_results"):
        os.system("rm -rf video_results")
    os.system("mkdir -p video_results")
    os.system("mkdir -p video_results/fgr")
    os.system("mkdir -p video_results/fgr/video0")


    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    fps = min(MAX_FPS, fps)

    os.system(f"ffmpeg -i {video} -vf fps={fps} video_results/fgr/video0/%04d.jpg")

    frame_names = sorted(os.listdir("video_results/fgr/video0"))
    if len(frame_names) > MAX_FRAMES:
        print(f"Too many frames ({len(frame_names)}), only process the first {MAX_FRAMES} frames.")
        for frame_name in frame_names[MAX_FRAMES:]:
            frame_path = os.path.join("video_results/fgr/video0", frame_name)
            os.remove(frame_path)
        frame_names = frame_names[:MAX_FRAMES]
    
    # Find human masks in the first 10 frame using Mask-RCNN
    masks = None
    mask_index = 0

    for i in range(min(10, len(frame_names)// fps)):
        frame_path = os.path.join(DEFAULT_FRAME_PATH, frame_names[i * fps])
        image = Image.open(frame_path)
        _, masks = predict_human_mask(image)

        # Save masks as png images
        if masks.max() > 0:
            mask_index = i
            break
    
    if masks.max() == 0:
        return
        
    for frame_name in frame_names[:mask_index]:
        frame_path = os.path.join(DEFAULT_FRAME_PATH, frame_name)
        os.remove(frame_path)
    
    frame_names = frame_names[mask_index:]

    # Process video with XMem
    torch.cuda.empty_cache()
    processor = InferenceCore(network, config=config)
    n_objects = masks.max()
    processor.set_all_labels(range(1, n_objects+1)) # consecutive labels
    current_frame_index = 0

    resized_ratio = 1.0
    ori_size = image.size   
    
    if max(image.size) > MAX_SIZE:
        resized_ratio = MAX_SIZE * 1.0 / max(image.size)

    # with torch.cuda.amp.autocast(enabled=True):
    with torch.no_grad():
        for frame_name in frame_names:
            # load frame-by-frame
            frame = cv2.imread(os.path.join("video_results/fgr/video0", frame_name))
            frame = cv2.resize(frame, (0, 0), fx=resized_ratio, fy=resized_ratio)

            # convert numpy array to pytorch tensor format
            frame_torch, _ = image_to_torch(frame, device="cuda")
            
            if current_frame_index == 0:
                # initialize with the mask
                resized_masks = cv2.resize(masks, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_torch = index_numpy_to_one_hot_torch(resized_masks, n_objects+1).to("cuda")
                # the background mask is not fed into the model
                prediction = processor.step(frame_torch, mask_torch[1:])
            else:
                # propagate only
                prediction = processor.step(frame_torch)

            if current_frame_index % 5 == 0:
                torch.cuda.empty_cache()

            # argmax, convert to numpy
            prediction = torch_prob_to_numpy_mask(prediction)
            prediction = cv2.resize(prediction, ori_size, interpolation=cv2.INTER_NEAREST)
            # save the mask
            for i in range(1, n_objects+1):
                mask = prediction == i
                mask = mask.astype("uint8") * 255
                out_path = f"video_results/mask/video0/" + frame_name.replace(".jpg", "") +  f"/{i-1:02d}.png"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, mask)

            current_frame_index += 1
    torch.cuda.empty_cache()

if __name__ == "__main__":
    video_path = sys.argv[1]
    process_video(video_path)