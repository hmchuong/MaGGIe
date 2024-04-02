import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

global out_dir, pha_dir, image_dir, bg_dir, valid_videos
split = 'train'
bg_dir = '/home/chuongh/vm2m/data/bg'
no_videos = 1000

image_dir = f'/home/chuongh/vm2m/data/VideoMatte240K/{split}/fgr'
pha_dir = f'/home/chuongh/vm2m/data/VideoMatte240K/{split}/clean_pha'
out_dir = f'/home/chuongh/vm2m/data/VideoMatte240K_syn/{split}'
valid_videos = []

def gen_video(out_name):
    random_state = np.random.RandomState(int(out_name))
    no_videos = random_state.randint(2, 3) if split != 'train' else random_state.randint(2, 5)
    video_names = random_state.choice(valid_videos, no_videos, replace=False)
    # Get random background
    while True:
        bg_name = random_state.choice(os.listdir(bg_dir))
        bg_path = os.path.join(bg_dir, bg_name)
        bg = cv2.imread(bg_path)[:, :, ::-1]
        if bg.shape[1] > bg.shape[0]:
            break


    # Get no.frames of each video --> min no.frames
    no_frames = []
    for video_name in video_names:
        fgr_path = os.path.join(image_dir, video_name)
        no_frames.append(len(os.listdir(fgr_path)))

    new_no_frames = min(no_frames)

    # Compute the (x, y, w, h) of fg in each video, random select start frame
    fg_bboxes = []
    start_frames = []
    for video_name in video_names:
        alpha_paths = os.listdir(os.path.join(pha_dir, video_name))
        alpha_paths = sorted(alpha_paths)
        if len(alpha_paths) == new_no_frames:
            start_frame = 0
        else:
            start_frame = random_state.randint(0, len(alpha_paths)-new_no_frames)
        start_frames.append(start_frame)
        x1, y1, x2, y2 = 999999, 999999, 0, 0
        for alpha_name in alpha_paths[start_frame:start_frame+new_no_frames]:
            # Read alpha
            alpha = Image.open(os.path.join(pha_dir, video_name, alpha_name)).convert('L')

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
    space_w = w // 2
    for box in fg_bboxes:
        ratio = 1.0
        if box[3] > 0.7 * h:
            ratio = random_state.uniform(0.7, 0.9) * h / box[3]
        if box[2] > space_w:
            ratio = min(ratio, random_state.uniform(0.8, 1.0) * space_w / box[2])
        resized_ratios.append(ratio)

    # Compute x, y for each video
    x = 0
    composited_bboxes = []
    for i in range(len(video_names)):
        box = fg_bboxes[i]
        ratio = resized_ratios[i]
        new_h = int(box[3] * ratio)
        new_w = int(box[2] * ratio)
        x1 = x + random_state.randint(0, new_w // 2) * random_state.choice([-1, 1])
        x1 = min(x1, w - new_w)
        y1 = random_state.randint(0, h - new_h)
        composited_bboxes.append((x1, y1, new_w, new_h))
        x = x1 + new_w


    # Combine the video: fgr and alpha
    for i in range(new_no_frames):
        new_image = Image.fromarray(bg)
        all_alphas = []
        for vid_idx in range(len(video_names)):
            video_name = video_names[vid_idx]
            start_frame = start_frames[vid_idx]
            alpha_names = os.listdir(os.path.join(pha_dir, video_name))
            alpha_names = sorted(alpha_names)
            alpha_name = alpha_names[start_frame: start_frame + new_no_frames][i]
            alpha = Image.open(os.path.join(pha_dir, video_name, alpha_name)).convert('L')

            fgr_names = os.listdir(os.path.join(image_dir, video_name))
            fgr_names = sorted(fgr_names)
            fgr_name = fgr_names[start_frame: start_frame + new_no_frames][i]
            fgr = Image.open(os.path.join(image_dir, video_name, fgr_name)).convert('RGB')

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
                all_alphas[j] = all_alphas[j] * (1 - all_alphas[-1])
        
        # TODO: Save video frame
        fgr_path = os.path.join(out_dir, 'fgr', out_name, f'{i:05d}.jpg')
        os.makedirs(os.path.dirname(fgr_path), exist_ok=True)
        new_image.save(fgr_path)

        for alpha_i, alpha in enumerate(all_alphas):
            alpha_path = os.path.join(out_dir, 'pha', out_name, f'{i:05d}', f'{alpha_i:02d}.png')
            os.makedirs(os.path.dirname(alpha_path), exist_ok=True)
            Image.fromarray((alpha * 255).astype('uint8')).save(alpha_path)



if __name__ == "__main__":
    if split == 'train':
        with open("/home/chuongh/vm2m/vm2m/dataloader/valid_vm240k_train.txt", "r") as f:
            for line in f:
                valid_videos.append(line.strip())
    elif split == 'test':
        valid_videos.extend(['vm_0000', 'vm_0002', 'vm_0003', 'vm_0004'])
    elif split == 'valid':
        valid_videos.extend(list(os.listdir(image_dir)))
    out_videos = []
    for i in range(no_videos):
        out_videos.append(f'{i:05d}')
    # import pdb; pdb.set_trace()
    with Pool(80) as p:
        pbar = tqdm(total=len(out_videos))
        for _ in p.imap_unordered(gen_video, out_videos):
            pbar.update()