import glob
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
try:
    from . import transforms as T
    from .utils import gen_transition_gt
except:
    import transforms as T
    from utils import gen_transition_gt

class HIMDataset(Dataset):
    def __init__(self, root_dir, split, padding_inst=10, short_size=768):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.short_size = short_size
        self.padding_inst = padding_inst
        self.load_image_alphas()

        self.transforms = [
            T.Load(),
            T.MasksFromBinarizedAlpha(),
            T.ResizeShort(short_size, transform_alphas=False),
            T.PaddingMultiplyBy(32, transform_alphas=False),
            T.Stack(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        self.transforms = T.Compose(self.transforms)
    
    def load_image_alphas(self):
        images = sorted(glob.glob(os.path.join(self.root_dir, "images", self.split, "*.jpg")))
        all_alphas = []
        for image in images:
            image_name = os.path.basename(image).replace(".jpg", "")
            alphas = sorted(glob.glob(os.path.join(self.root_dir, "alphas", self.split, image_name, "*.png")))
            all_alphas.append(alphas)
        self.data = list(zip(images, all_alphas))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, alphas = self.data[index]

        # Load image
        input_dict = {
            "frames": [image],
            "alphas": alphas,
            "masks": None
        }
        output_dict = self.transforms(input_dict)
        image, alpha, mask, transform_info = output_dict["frames"], output_dict["alphas"], output_dict["masks"], output_dict["transform_info"]
        
        add_padding = self.padding_inst - len(alphas)
        if add_padding > 0:
            alpha = torch.cat([alpha, torch.zeros(add_padding,*alpha.shape[1:])], dim=0)
            mask = torch.cat([mask, torch.zeros(add_padding, *mask.shape[1:])], dim=0)

        mask = F.interpolate(mask, size=(mask.shape[2] // 8, mask.shape[3] // 8), mode="nearest")
        alpha = alpha * 1.0 / 255
        mask = mask * 1.0 / 255

        out =  {'image': image, 
                'mask': mask.float()[None, :, 0], 
                'alpha': alpha.float()[None, :, 0]}
        
        # Generate trimap for evaluation
        trans = gen_transition_gt(alpha)
        trimap = torch.zeros_like(alpha)
        trimap[alpha > 0.5] = 2.0 # FG
        trimap[trans > 0] = 1.0 # Transition
        out.update({'trimap': trimap[None, :, 0], 'image_names': [image], 'transform_info': transform_info, "skip": 0})

        return out

if __name__ == "__main__":
    import cv2
    import numpy as np
    dataset = HIMDataset(root_dir="/mnt/localssd/HIM2K", split="comp")
    for batch in dataset:
        frames, masks, alphas, transition_gt = batch["image"], batch["mask"], batch["alpha"], batch["trimap"]
        frame = frames[0] * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        frame = (frame * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        cv2.imwrite("frame.png", frame[:, :, ::-1])
        for idx in range(masks.shape[1]):
            mask = masks[0, idx]
            alpha = alphas[0, idx]
            transition = transition_gt[0, idx]
            cv2.imwrite("mask_{}.png".format(idx), mask.numpy() * 255)
            cv2.imwrite("alpha_{}.png".format(idx), alpha.numpy() * 255)
            cv2.imwrite("trimap_{}.png".format(idx), transition.numpy() * 80)
        import pdb; pdb.set_trace()

