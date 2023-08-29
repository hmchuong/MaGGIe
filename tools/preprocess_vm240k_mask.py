import os
import glob
import cv2
import numpy as np
from PIL import Image
from skimage.measure import label as sklabel
from tqdm import tqdm
from multiprocessing import Pool

global out_dir, pha_dir
split = 'train'
pha_dir = f'/home/chuongh/vm2m/data/VideoMatte240K/{split}/pha'
out_dir = f'/home/chuongh/vm2m/data/VideoMatte240K/{split}/clean_pha'

def process_alpha(pha_path):
    out_path = pha_path.replace(pha_dir, out_dir).replace(".jpg", ".png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Read alpha
    alpha_ori = Image.open(pha_path).convert('L')
    alpha_ori = np.array(alpha_ori)

    # Threshold alpha > 1.0/255.0
    alpha = alpha_ori > 1.0
    alpha = alpha.astype(np.uint8)

    # Find connected components
    labels, num = sklabel(alpha, return_num=True)
    if num > 1:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        alpha_ori = alpha_ori * largestCC
    # Find the largest connected component
    # areas = []
    # for i in range(1, num+1):
    #     areas.append(np.sum(labels==i))
    # max_area = np.argmax(areas) + 1

    # Find the bounding box of the largest connected component
    # max_area_mask = (labels == max_area)
    # alpha_ori[~max_area_mask] = 0
    
    Image.fromarray(alpha_ori).save(out_path)

if __name__ == "__main__":
    pha_paths = list(glob.glob(os.path.join(pha_dir, '*/*.jpg')))
    # process_alpha(pha_paths[0])
    with Pool(80) as p:
        pbar = tqdm(total=len(pha_paths))
        for _ in p.imap_unordered(process_alpha, pha_paths):
            pbar.update()