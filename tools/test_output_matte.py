import os
import argparse
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from vm2m.utils.metric import build_metric
from vm2m.dataloader.utils import gen_transition_gt

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--split', type=str, default='natural')
    argparser.add_argument('--pred', type=str, required=True)
    argparser.add_argument('--trimap', action='store_true')
    args = argparser.parse_args()

    val_error_dict = build_metric(['MAD', 'MSE', 'SAD'])

    gt_dir = f'/mnt/localssd/HIM2K/alphas/{args.split}'
    pred_dir = args.pred

    for image_name in tqdm(os.listdir(gt_dir)):
        image_dir = os.path.join(gt_dir, image_name)
        for mask_name in os.listdir(image_dir):
            gt_mask_path = os.path.join(image_dir, mask_name)
            pred_mask_path = gt_mask_path.replace(gt_dir, pred_dir)
            if not os.path.exists(pred_mask_path):
                continue
            gt_mask = Image.open(gt_mask_path).convert('L')
            pred_mask = Image.open(pred_mask_path).convert('L')
            gt_mask = np.array(gt_mask) / 255.0
            pred_mask = np.array(pred_mask) / 255.0
            
            trimap = None
            if args.trimap:
                trans = gen_transition_gt(torch.from_numpy(gt_mask[None, None])).numpy().squeeze()
                trimap = np.zeros_like(gt_mask)
                trimap[gt_mask > 0.5] = 2.0 # FG
                trimap[trans > 0] = 1.0 # Transition

            for k, v in val_error_dict.items():
                v.update(pred_mask, gt_mask, trimap=trimap)
    
    for k, v in val_error_dict.items():
        print("{}: {}\n".format(k, v.average()))
