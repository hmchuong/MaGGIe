import os
import argparse
import cv2
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
from multiprocessing import Pool

global valid_videos, clip_len, bg_files, bg_videos, data_dir, out_dir, level
split = 'train'
no_videos = 300
clip_len = 30
data_dir = '/home/chuongh/vm2m/data/VHM'
valid_videos = []
bg_files = []
bg_videos = []
out_dir = ''
level = "hard"

def load_alpha_paths(video_name):
    alpha_dir = os.path.join(data_dir, video_name).replace("/fgr/", "/pha/")
    if not os.path.exists(alpha_dir):
        alpha_dir = os.path.join(data_dir, video_name)
    alpha_paths = os.listdir(alpha_dir)
    alpha_paths = sorted(alpha_paths)
    alpha_paths = [os.path.join(alpha_dir, alpha_path) for alpha_path in alpha_paths]
    return alpha_paths

def gen_video(out_name):
    random_state = np.random.RandomState(int(out_name))
    no_videos = random_state.randint(2, 5)
    if level == "easy":
        no_videos = min(no_videos, 3)
    video_names = random_state.choice(valid_videos, no_videos, replace=False)
    
    # Get random background
    bg_paths = []
    bg = None
    if random_state.rand() < 0.5:
        while True:
            bg_name = random_state.choice(bg_files)
            bg_path = os.path.join(data_dir, bg_name)
            bg = cv2.imread(bg_path)[:, :, ::-1]
            if bg.shape[1] > bg.shape[0]:
                break
        bg_paths = [bg_path]
    else:
        bg_name = random_state.choice(bg_videos)
        bg_paths = sorted(os.listdir(os.path.join(data_dir, bg_name)))
        bg_paths = [os.path.join(data_dir, bg_name, bg_path) for bg_path in bg_paths]
        bg = cv2.imread(bg_paths[0])[:, :, ::-1]

    # Get no.frames of each video --> min no.frames
    no_frames = []
    for video_name in video_names:
        fgr_path = os.path.join(data_dir, video_name)
        no_frames.append(len(os.listdir(fgr_path)))

    new_no_frames = min(min(no_frames), clip_len)

    # Compute the (x, y, w, h) of fg in each video, random select start frame
    fg_bboxes = []
    start_frames = []
    for video_name in video_names:
        alpha_paths = load_alpha_paths(video_name)

        if len(alpha_paths) == new_no_frames:
            start_frame = 0
        else:
            start_frame = random_state.randint(0, len(alpha_paths)-new_no_frames)
        start_frames.append(start_frame)
        x1, y1, x2, y2 = 999999, 999999, 0, 0
        for alpha_name in alpha_paths[start_frame:start_frame+new_no_frames]:
            # Read alpha
            alpha = Image.open(alpha_name).convert('L')

            # Threshold alpha > 1.0/255.0
            alpha = np.array(alpha).astype(np.uint8)
            box = cv2.boundingRect(alpha)
            box = list(box)
            box[2] += box[0]
            box[3] += box[1]
            x1 = min(x1, box[0])
            y1 = min(y1, box[1])
            x2 = max(x2, box[2])
            y2 = max(y2, box[3])
            
        fg_bboxes.append((x1, y1, x2 - x1, y2 - y1))

    # Compute the resize ratio with regarding to the background size: height = 70-90% of bg height
    h, w = bg.shape[:2]
    resized_ratios = []
    # target_h = random_state.uniform(0.7, 0.9) * h
    instance_w_over_h = [box[2] * 1.0 / box[3] for box in fg_bboxes]
    
    for box, w_over_h in zip(fg_bboxes, instance_w_over_h):
        target_w = w * (w_over_h / sum(instance_w_over_h))
        ratio = target_w / box[2]
        # Change ratio randomly
        if level == "medium":
            ratio = ratio * random_state.uniform(1.0, 1.5)
        elif level == "hard":
            ratio = ratio * random_state.uniform(1.0, 2.0)
        if ratio * box[3] > h:
            ratio = h / box[3] * random_state.uniform(0.8, 1.0)
        resized_ratios.append(ratio)

    # Compute x, y for each video
    x = 0
    composited_bboxes = []
    for i in range(len(video_names)):
        box = fg_bboxes[i]
        ratio = resized_ratios[i]
        new_h = int(box[3] * ratio)
        new_w = int(box[2] * ratio)
        if level == "easy":
            x1 = x
        elif level == "medium" or level == "hard":
            x1 = x + random_state.randint(0, w // 2) * random_state.choice([-1, 1])
        # x1 = x #+ random_state.randint(0, new_w // 2) * random_state.choice([-1, 1])
        x1 = min(x1, w - new_w)
        x1 = max(x1, 0)
        # y1 = random_state.randint(0, h - new_h)
        y1 = h - new_h
        composited_bboxes.append((x1, y1, new_w, new_h))
        x = x1 + new_w


    # Combine the video: fgr and alpha
    start_bg_frame = random_state.randint(0, len(bg_paths) - 1) if len(bg_paths) > 1 else 0
    max_occluded = 0
    for i in range(new_no_frames):
        # Load background
        bg_path = bg_paths[min(start_bg_frame + i, len(bg_paths) - 1)]
        new_image = Image.open(bg_path).convert('RGB')
        # new_image = Image.fromarray(bg)
        all_alphas = []
        for vid_idx in range(len(video_names)):
            video_name = video_names[vid_idx]
            start_frame = start_frames[vid_idx]
            
            alpha_names = load_alpha_paths(video_name)
            alpha_name = alpha_names[start_frame: start_frame + new_no_frames][i]
            if "/fgr/" in alpha_name:
                alpha = Image.open(alpha_name)
                alpha = np.array(alpha)[:, :, 3]
                alpha = Image.fromarray(alpha)
            else:
                alpha = Image.open(alpha_name).convert('L')

            fgr_names = os.listdir(os.path.join(data_dir, video_name))
            fgr_names = sorted(fgr_names)
            fgr_name = fgr_names[start_frame: start_frame + new_no_frames][i]
            fgr = Image.open(os.path.join(data_dir, video_name, fgr_name)).convert('RGB')

            # Crop and resize fg and alpha
            box = fg_bboxes[vid_idx]
            ratio = resized_ratios[vid_idx]
            new_h = int(box[3] * ratio)
            new_w = int(box[2] * ratio)
            alpha = alpha.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
            fgr = fgr.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
            alpha = alpha.resize((new_w, new_h), Image.BILINEAR)
            fgr = fgr.resize((new_w, new_h), Image.BILINEAR)

            # Blend fg and new_image
            x1, y1, new_w, new_h = composited_bboxes[vid_idx]
            new_image.paste(fgr, (x1, y1), alpha)

            new_alpha = Image.new('L', (w, h), 0)
            new_alpha.paste(alpha, (x1, y1))

            all_alphas.append(np.array(new_alpha)/ 255.0)
            for j in range(len(all_alphas) - 1):
                new_alpha_j = all_alphas[j] * (1 - all_alphas[-1])
                if all_alphas[j].sum() > 0:
                    occluded = 1.0 - (new_alpha_j.sum() * 1.0 / (all_alphas[j].sum() + 1e-7))
                    if (occluded > 0.3 and level == "medium") or (occluded > 0.85 and level == "hard"):
                        shutil.rmtree(os.path.join(out_dir, 'fgr', out_name), ignore_errors=True)
                        shutil.rmtree(os.path.join(out_dir, 'pha', out_name), ignore_errors=True)
                        return
                    max_occluded = max(max_occluded, occluded)
                all_alphas[j] = new_alpha_j
        
        # Save video frame
        fgr_path = os.path.join(out_dir, 'fgr', out_name, f'{i:05d}.jpg')
        os.makedirs(os.path.dirname(fgr_path), exist_ok=True)
        new_image.save(fgr_path)

        for alpha_i, alpha in enumerate(all_alphas):
            alpha_path = os.path.join(out_dir, 'pha', out_name, f'{i:05d}', f'{alpha_i:02d}.png')
            os.makedirs(os.path.dirname(alpha_path), exist_ok=True)
            Image.fromarray((alpha * 255).astype('uint8')).save(alpha_path)
    
    if max_occluded < 0.05 and level == "medium":
        shutil.rmtree(os.path.join(out_dir, 'fgr', out_name), ignore_errors=True)
        shutil.rmtree(os.path.join(out_dir, 'pha', out_name), ignore_errors=True)
    if max_occluded < 0.5 and level == "hard":
        shutil.rmtree(os.path.join(out_dir, 'fgr', out_name), ignore_errors=True)
        shutil.rmtree(os.path.join(out_dir, 'pha', out_name), ignore_errors=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--data-dir', type=str, default='/home/chuongh/vm2m/data/VHM')
    parser.add_argument('--out-dir', type=str, default='/home/chuongh/vm2m/data/VHM/new_syn')
    parser.add_argument('--n-workers', type=int, default=80)
    args = parser.parse_args()

    split = args.split
    data_dir = args.data_dir
    out_dir = args.out_dir
    num_workers = args.n_workers

    # Load video dirs
    with open(os.path.join(data_dir, f'fg_{split}.txt'), 'r') as f:
        valid_videos.extend([line.strip() for line in f.readlines()])
    
    bg_txt = os.path.join(data_dir, f'bg_{split}.txt')
    with open(bg_txt, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.endswith(".jpg"):
                bg_files.append(line)
            else:
                bg_videos.append(line)
    
    if split == "train":
        # Easy cases
        no_videos = 500
        level = "easy"
        out_videos = []
        start_idx = 0
        for i in range(start_idx, no_videos + start_idx):
            out_videos.append(f'{i:05d}')
        with Pool(num_workers) as p:
            pbar = tqdm(total=len(out_videos))
            for _ in p.imap_unordered(gen_video, out_videos):
                pbar.update()
        
        # Medium cases
        no_videos = 1000
        level = "medium"
        out_videos = []
        start_idx = 500
        for i in range(start_idx, no_videos + start_idx):
            out_videos.append(f'{i:05d}')
        with Pool(num_workers) as p:
            pbar = tqdm(total=len(out_videos))
            for _ in p.imap_unordered(gen_video, out_videos):
                pbar.update()

        # Hard cases
        no_videos = 2000
        level = "hard"
        out_videos = []
        start_idx = 1500
        for i in range(start_idx, no_videos + start_idx):
            out_videos.append(f'{i:05d}')
        with Pool(num_workers) as p:
            pbar = tqdm(total=len(out_videos))
            for _ in p.imap_unordered(gen_video, out_videos):
                pbar.update()
        
        no_videos = 6000
        level = "medium"
        out_videos = []
        start_idx = 3500
        for i in range(start_idx, no_videos + start_idx):
            out_videos.append(f'{i:05d}')
        with Pool(num_workers) as p:
            pbar = tqdm(total=len(out_videos))
            for _ in p.imap_unordered(gen_video, out_videos):
                pbar.update()
    else:
        out_videos = []
        level = "easy"
        no_videos = 51
        for i in range(no_videos):
            out_videos.append(f'{i:05d}')
        with Pool(num_workers) as p:
            pbar = tqdm(total=len(out_videos))
            for _ in p.imap_unordered(gen_video, out_videos):
                pbar.update()

        out_videos = []
        level = "medium"
        for i in range(29, 222):
            out_videos.append(f'{i:05d}')
        with Pool(num_workers) as p:
            pbar = tqdm(total=len(out_videos))
            for _ in p.imap_unordered(gen_video, out_videos):
                pbar.update()

        out_videos = []
        level = "hard"
        for i in range(2, 81):
            out_videos.append(f'{i:05d}')
        with Pool(num_workers) as p:
            pbar = tqdm(total=len(out_videos))
            for _ in p.imap_unordered(gen_video, out_videos):
                pbar.update()