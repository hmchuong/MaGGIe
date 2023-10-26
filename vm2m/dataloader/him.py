import glob
import os
import logging
import numpy as np
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


def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x, factor=2):
    return x[:, :, ::factor, ::factor]

def upsample(x, size=2):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel.to(img.device), groups=img.shape[1])
    return out

def gaussian_laplacian_pyramid(img, kernel, loi=[1,4,8]):
    pyr_gau = [img]
    pyr_lap = []
    for i in range(1, len(loi)):
        filtered = conv_gauss(pyr_gau[-1], kernel)
        scale_factor = int(loi[i] / loi[i-1])
        down = downsample(filtered, factor=scale_factor)
        up = upsample(down, size=pyr_gau[-1].shape[-2:])
        diff = pyr_gau[-1]-up
        pyr_lap.append(diff)
        pyr_gau.append(down)
    pyr_lap.append(pyr_gau[-1])
    return pyr_gau, pyr_lap

class HIMDataset(Dataset):
    def __init__(self, root_dir, split, padding_inst=10, short_size=768, is_train=False, random_seed=2023, \
                crop=(512, 512), flip_p=0.5, bin_alpha_max_k=30, downscale_mask=True, alpha_dir_name='alphas', mask_dir_name='', 
                modify_mask_p=0.1, downscale_mask_p=0.5, use_maskrcnn_p=0.3, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.short_size = short_size
        self.padding_inst = padding_inst
        self.is_train = is_train
        self.downscale_mask = downscale_mask
        self.alpha_dir_name = alpha_dir_name
        self.mask_dir_name = mask_dir_name
        self.use_maskrcnn_p = use_maskrcnn_p
        self.random = np.random.RandomState(random_seed)
        if "HIM" in root_dir:
            self.load_him_image_alphas()
        elif "HHM" in root_dir:
            self.load_hhm_sythetic_image_alphas()
        

        # self.transforms = [
        #     T.Load(),
        #     T.ResizeShort(short_size, transform_alphas=False),
        #     T.PaddingMultiplyBy(32, transform_alphas=False),
        #     T.Stack(),
        #     T.MasksFromBinarizedAlpha() if is_train else T.RandomBinarizeAlpha(self.random, 30),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ]
        # if is_train: 
        #     self.transforms = self.transforms[:3] + [T.RandomCropByAlpha(crop, self.random)] + self.transforms[3:]

        self.transforms = [
            T.Load(),
            T.ResizeShort(short_size, transform_alphas=is_train),
            T.PaddingMultiplyBy(32, transform_alphas=is_train),
            T.Stack()
        ]
        if self.is_train:
            # self.transforms += [
            #     T.RandomCropByAlpha(crop, self.random),
            #     T.RandomHorizontalFlip(self.random, flip_p),
            #     T.ChooseOne(self.random, [
            #         T.ModifyMaskBoundary(self.random, modify_mask_p),
            #         T.Compose([
            #             T.RandomBinarizedMask(self.random, bin_alpha_max_k),
            #             T.DownUpMask(self.random, 0.125, downscale_mask_p)
            #         ])
            #     ])
                
            # ]
            self.transforms += [
                T.RandomCropByAlpha(crop, self.random),
                T.RandomHorizontalFlip(self.random, flip_p),
                T.Compose([
                    T.RandomBinarizedMask(self.random, bin_alpha_max_k),
                    T.DownUpMask(self.random, 0.125, downscale_mask_p)
                ])
            ]
        else:
            if self.mask_dir_name == '':
                self.transforms += [T.GenMaskFromAlpha(), T.DownUpMask(self.random, 0.125, 1.0)]
        self.transforms += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        self.transforms = T.Compose(self.transforms)


        # For gaussian laplacian pyramid
        self.image_kernel = gauss_kernel(channels=3)
    
    def load_him_image_alphas(self):
        images = sorted(glob.glob(os.path.join(self.root_dir, "images", self.split, "*.jpg")))
        all_alphas = []
        target_dir_name = self.alpha_dir_name if self.mask_dir_name == '' else self.mask_dir_name
        valid_images = []
        for image in images:
            image_name = os.path.basename(image).replace(".jpg", "")
            alpha_dir = os.path.join(self.root_dir, target_dir_name, self.split, image_name)
            if not os.path.exists(alpha_dir):
                continue
            valid_images.append(image)
            alphas = sorted(os.listdir(alpha_dir))
            all_alphas.append([os.path.join(self.root_dir, self.alpha_dir_name, self.split, image_name, p) for p in alphas])
        
        self.data = list(zip(valid_images, all_alphas))
    
    def load_hhm_sythetic_image_alphas(self):
        images = sorted(glob.glob(os.path.join(self.root_dir, self.split, "images", "*.jpg")))
        all_alphas = []
        for image in images:
            image_name = os.path.basename(image).replace(".jpg", "")
            alphas = sorted(glob.glob(os.path.join(self.root_dir, self.split, self.alpha_dir_name, image_name, "*.png")))
            all_alphas.append(alphas)
        self.data = list(zip(images, all_alphas))
    
    def __len__(self):
        return len(self.data)
    
    def load_masks_train(self, alphas):
        masks = []
        for alpha in alphas:
            mask_path = alpha.replace(self.alpha_dir_name, self.mask_dir_name)
            # TODO: Change the prob here
            if os.path.exists(mask_path) and self.random.rand() < self.use_maskrcnn_p:
                masks.append(mask_path)
            else:
                masks.append(alpha)
        return masks
    
    def __getitem__(self, index):
        image_path, alphas = self.data[index]

        if len(alphas) > self.padding_inst:
            alphas = self.random.choice(alphas, self.padding_inst, replace=False)

        # remove one random alpha
        if len(alphas) > 1 and self.is_train and self.random.rand() < 0.2:
            num_alphas = len(alphas) - 1
            alphas = self.random.choice(alphas, num_alphas, replace=False)

        # Load mask path and random replace the mask by alpha
        masks = None
        if self.is_train:
            masks = self.load_masks_train(alphas)
        elif self.mask_dir_name != '':
            masks = [alpha.replace(self.alpha_dir_name, self.mask_dir_name) for alpha in alphas]

        

        # Load image
        input_dict = {
            "frames": [image_path],
            "alphas": alphas,
            "masks": masks,
            "weights": None
        }
        output_dict = self.transforms(input_dict)
        image, alpha, mask, transform_info = output_dict["frames"], output_dict["alphas"], output_dict["masks"], output_dict["transform_info"]
        
        if not self.is_train:
            alpha = output_dict["ori_alphas"]
        if mask.sum() == 0:
            logging.warning("Mask is empty")
            return self.__getitem__(self.random.randint(0, len(self.data)))
        alpha = alpha * 1.0 / 255
        mask = mask * 1.0 / 255
        add_padding = self.padding_inst - len(alphas)
        if add_padding > 0 and self.is_train:
            new_alpha = torch.zeros(1, self.padding_inst, *alpha.shape[2:])
            new_mask = torch.zeros(1, self.padding_inst, *mask.shape[2:])
            chosen_ids = self.random.choice(range(self.padding_inst), len(alphas), replace=False)
            new_alpha[:, chosen_ids] = alpha
            new_mask[:, chosen_ids] = mask
            mask = new_mask
            alpha = new_alpha
            # alpha = torch.cat([alpha, torch.zeros(add_padding,*alpha.shape[1:])], dim=0)
            # mask = torch.cat([mask, torch.zeros(add_padding, *mask.shape[1:])], dim=0)
        if self.downscale_mask:
            mask = F.interpolate(mask, size=(image.shape[2] // 8, image.shape[3] // 8), mode="nearest")
        # alpha = alpha * 1.0 / 255
        # mask = mask * 1.0 / 255
        # import pdb; pdb.set_trace()

        # Build Gauss OS8, Lap OS4, Lap OS1, Lap Image OS1-OS8
        _, im_pyrlap = gaussian_laplacian_pyramid(image, self.image_kernel, loi=[1,8])
        alpha_kernel = gauss_kernel(channels=alpha.shape[1])
        _, alpha_pyrlap = gaussian_laplacian_pyramid(alpha, alpha_kernel, loi=[1,4,8])

        out =  {
            'image': image, 
            'image_lap': im_pyrlap[0].abs(),
            'mask': mask.float(),
            'alpha': alpha.float(),
            'alpha_gt_os1': alpha_pyrlap[0],
            'alpha_gt_os4': alpha_pyrlap[1],
            'alpha_gt_os8': alpha_pyrlap[2],
        }
        # import pdb; pdb.set_trace()
        # Generate trimap for evaluation
        if self.is_train:
            k_size = self.random.choice(range(2, 5))
            iterations = self.random.randint(5, 15)
            trans = gen_transition_gt(alpha[0, :, None], mask[0, :, None], k_size=k_size, iterations=iterations)
            out.update({'transition': trans.float()[None, :, 0]})
        else:
            trans = gen_transition_gt(alpha[0, :, None])
            trans = trans.squeeze(1)[None]
            trimap = torch.zeros_like(alpha)
            trimap[alpha > 0.5] = 2.0 # FG
            trimap[trans > 0] = 1.0 # Transition
            out.update({'trimap': trimap, 'image_names': [image_path], 'alpha_names': [os.path.basename(a) for a in alphas], 'transform_info': transform_info, "skip": 0})
            
        return out

if __name__ == "__main__":
    import cv2
    import numpy as np
    dataset = HIMDataset(root_dir="/mnt/localssd/HHM", split="synthesized", is_train=True, downscale_mask=False)
    # dataset = HIMDataset(root_dir="/mnt/localssd/HIM2K", split="natural", is_train=False, downscale_mask=False, mask_dir_name='masks_matched')
    for batch in dataset:
        # batch = dataset[117]
        frames, masks, alphas, transition_gt = batch["image"], batch["mask"], batch["alpha"], batch.get("transition", batch.get("trimap"))
        image_lap = batch["image_lap"]
        alpha_gt_os1 = batch["alpha_gt_os1"]
        alpha_gt_os4 = batch["alpha_gt_os4"]
        alpha_gt_os8 = batch["alpha_gt_os8"]
        frame = frames[0] * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        frame = (frame * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        image_lap = image_lap[0] * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image_lap = (image_lap * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        cv2.imwrite("debug/frame.png", frame[:, :, ::-1])
        cv2.imwrite("debug/image_lap.png", image_lap[:, :, ::-1])
        
        for idx in range(masks.shape[1]):
            mask = masks[0, idx]
            alpha = alphas[0, idx]
            transition = transition_gt[0, idx]
            cv2.imwrite("debug/mask_{}.png".format(idx), mask.numpy() * 255)
            cv2.imwrite("debug/alpha_{}.png".format(idx), alpha.numpy() * 255)
            cv2.imwrite("debug/alpha_gt_os1_{}.png".format(idx), alpha_gt_os1[0, idx].numpy() * 255)
            cv2.imwrite("debug/alpha_gt_os4_{}.png".format(idx), alpha_gt_os4[0, idx].numpy() * 255)
            cv2.imwrite("debug/alpha_gt_os8_{}.png".format(idx), alpha_gt_os8[0, idx].numpy() * 255)
            cv2.imwrite("debug/trimap_{}.png".format(idx), transition.numpy() * 80)
        import pdb; pdb.set_trace()

