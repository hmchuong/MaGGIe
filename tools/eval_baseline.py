import sys
import os
import glob
sys.path.append('..')
import copy
import cv2
import tqdm
import numpy as np
import torch
from vm2m.utils.metric import build_metric
from vm2m.dataloader import MultiInstVidDataset


pred_dir = sys.argv[1]
subset = sys.argv[2]
is_fast = os.getenv("FAST_EVAL", "0") == "1"

# create dataset and dataloader
def evaluate(subset):
    dataset = MultiInstVidDataset(root_dir="/mnt/localssd/syn/benchmark", split=subset, clip_length=1, overlap=0, padding_inst=10, is_train=False, short_size=576, 
                        crop=[512, 512], flip_p=0.5, bin_alpha_max_k=30,
                        max_step_size=5, random_seed=2023, mask_dir_name='mask', pha_dir='pha', weight_mask_dir='', is_ss_dataset=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    val_error_dict = build_metric(['MAD', 'MSE', 'SAD', 'Grad', 'Conn', 'dtSSD', 'MESSDdt']) 
    if is_fast:
        val_error_dict = build_metric(['MAD', 'MSE', 'SAD', 'dtSSD']) 
    val_error_dict["MAD_fg"] = copy.deepcopy(val_error_dict['MAD'])
    val_error_dict["MAD_bg"] = copy.deepcopy(val_error_dict['MAD'])
    val_error_dict["MAD_unk"] = copy.deepcopy(val_error_dict['MAD'])

    # evaluate the model
    video_name = None
    all_preds = []
    all_gts = []
    all_trimap = []
    for i, batch in tqdm.tqdm(enumerate(dataloader)):
        image_names = batch.pop('image_names')
        trimap = batch.pop('trimap').numpy()
        alpha_gt = batch.pop('alpha').numpy()
        
        if image_names[0][0].split('/')[-2] != video_name:
            if len(all_gts) > 0:
                all_preds = np.stack(all_preds, axis=0)
                all_trimap = np.concatenate(all_trimap, axis=0)
                all_gts = np.concatenate(all_gts, axis=0)

                current_metrics = {}
                for k, v in val_error_dict.items():
                    current_trimap = None
                    if k.endswith("_fg"):
                        current_trimap = (all_trimap[None] == 2).astype('float32')
                    elif k.endswith("_bg"):
                        current_trimap = (all_trimap[None] == 0).astype('float32')
                    elif k.endswith("_unk"):
                        current_trimap = (all_trimap[None] == 1).astype('float32')
                    current_metrics[k] = v.update(all_preds[None], all_gts[None], trimap=current_trimap)
                
                log_str = f"{video_name}: "
                for k, v in current_metrics.items():
                    log_str += "{} - {:.4f}, ".format(k, v)
                all_preds = []
                all_gts = []
                all_trimap = []
                print(video_name, log_str)
            video_name = image_names[0][0].split('/')[-2]

        all_gts.append(alpha_gt[0])
        all_trimap.append(trimap[0])
        
        # TODO: load the prediction
        video_name, frame_name = image_names[0][0].split('/')[-2:]
        mask_names = glob.glob(os.path.join(pred_dir, subset, video_name, frame_name.replace(".jpg", "")) + '/*.png')
        mask_names = sorted(mask_names)
        if len(mask_names) == 0:
            mask_names = [os.path.join(pred_dir, subset, video_name, frame_name.replace(".jpg", ".png"))]
        all_masks = []
        for mask_name in mask_names:
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (alpha_gt.shape[-1], alpha_gt.shape[-2]))
            mask = mask / 255.0
            all_masks.append(mask)
        all_masks = np.stack(all_masks, axis=0)
        all_preds.append(all_masks)

    all_preds = np.stack(all_preds, axis=0)
    all_trimap = np.concatenate(all_trimap, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    current_metrics = {}
    for k, v in val_error_dict.items():
        current_trimap = None
        if k.endswith("_fg"):
            current_trimap = (all_trimap[None] == 2).astype('float32')
        elif k.endswith("_bg"):
            current_trimap = (all_trimap[None] == 0).astype('float32')
        elif k.endswith("_unk"):
            current_trimap = (all_trimap[None] == 1).astype('float32')
        current_metrics[k] = v.update(all_preds[None], all_gts[None], trimap=current_trimap)

    log_str = f"{video_name}: "
    for k, v in current_metrics.items():
        log_str += "{} - {:.4f}, ".format(k, v)
    print(video_name, log_str)

    print("Metrics:")
    metric_str = ""
    plain_str = ""
    for k, v in val_error_dict.items():
        metric_str += "{}: {}\n".format(k, v.average())
        plain_str += str(v.average()) + ","
    print(metric_str)
    print(subset)
    print(plain_str)

evaluate(subset)