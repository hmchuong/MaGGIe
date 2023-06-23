import cv2
import torch
import numpy as np

def gen_transition_gt(alphas, masks=None, k_size=25, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (k_size, k_size))
    all_trans_map = []
    for x in alphas:
        dilated = cv2.dilate(x[0, :, :, None].numpy(), kernel, iterations=iterations)
        eroded = cv2.erode(x[0, :, :, None].numpy(), kernel, iterations=iterations)
        trans_map = ((dilated - eroded) > 0).astype(float)
        all_trans_map.append(torch.from_numpy(trans_map))
    all_trans_map = torch.stack(all_trans_map).unsqueeze(1)
    
    if masks is not None:
        upmasks = torch.repeat_interleave(masks, 8, dim=-1)
        upmasks = torch.repeat_interleave(upmasks, 8, dim=-2)
        diff = (alphas > 127) != (upmasks == 255)
        all_trans_map[diff > 0] = 1.0
    
    return all_trans_map