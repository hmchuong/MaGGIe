import cv2
import os
from PIL import Image
import glob
import numpy as np
import sys
sys.path.append('../')
from tqdm import tqdm

from vm2m.dataloader.transforms import ModifyMaskBoundary, RandomBinarizedMask
from multiprocessing import Pool

global alpha_paths


def process_mask(index):
    random = np.random.RandomState(index)
    modily_mask_boundary = ModifyMaskBoundary(random, p=0.5)
    random_binary_mask = RandomBinarizedMask(random)
    alpha_path = alpha_paths[index]#[index // 10]
    # for i in range(10):
    i = index % 10
    out_dir = "mask" #+ str(i)
    alpha = np.array(Image.open(alpha_path).convert("L"))
    new_masks = ((alpha > 127) * 255).astype('uint8')
    input_dict = {}
    input_dict['masks'] = new_masks[None]
    if random.rand() < 0.5:
        input_dict = modily_mask_boundary(input_dict)
    else:
        input_dict = random_binary_mask(input_dict)
    new_masks = input_dict['masks'][0]
    new_masks = Image.fromarray(new_masks)
    out_path = alpha_path.replace("pha", out_dir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new_masks.save(out_path)

alpha_path = "/home/chuongh/vm2m/data/VHM/benchmark"
# import pdb; pdb.set_trace()
alpha_paths = glob.glob(alpha_path + "/*/pha/*/*/*.png")

with Pool(80) as p:
    pbar = tqdm(total=len(alpha_paths) * 1)
    for _ in tqdm(p.imap_unordered(process_mask, range(len(alpha_paths) * 1))):
        pbar.update()
