import os
import numpy as np
import cv2
from PIL import Image
import tqdm
import shutil
from multiprocessing import Pool
global iou_threshold, alpha_dir, mask_dir, out_dir
alpha_dir = '/home/chuongh/vm2m/data/HHM/synthesized/masks'
mask_dir = '/home/chuongh/vm2m/data/HHM/synthesized/alphas'
out_dir = '/home/chuongh/vm2m/data/HHM/synthesized/masks_matched'
iou_threshold = 0.7
def process(image_name):
    image_dir = os.path.join(alpha_dir, image_name)
    all_alphas = []
    all_names = []
    for alpha_name in os.listdir(image_dir):
        alpha = np.array(Image.open(os.path.join(image_dir, alpha_name)).convert('L')) / 255.0
        all_alphas.append(alpha)
        all_names.append(alpha_name)
    image_dir = os.path.join(mask_dir, image_name)
    for mask_name in os.listdir(image_dir):
        mask = np.array(Image.open(os.path.join(image_dir, mask_name)).convert('L')) / 255.0
        mask = mask > 0.0
        match_idx = -1
        max_iou = 0
        for i in range(len(all_alphas)):
            alpha = all_alphas[i]
            iou = np.sum(np.minimum(alpha, mask)) / (np.sum(np.maximum(alpha, mask)) + 1e-6)
            if iou > max_iou:
                max_iou = iou
                match_idx = i
        if max_iou > iou_threshold:
            # os.rename(os.path.join(image_dir, mask_name), os.path.join(image_dir, all_names[match_idx]))
            target_path = os.path.join(out_dir, image_name, mask_name)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(os.path.join(alpha_dir, image_name, all_names[match_idx]), target_path)

if __name__ == "__main__":
    image_names = os.listdir(alpha_dir)
    with Pool(80) as p:
        pbar = tqdm.tqdm(total=len(image_names))
        for _ in p.imap_unordered(process, image_names):
            pbar.update(1)