import os
import glob
import logging
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
try:
    from . import transforms as T
    from .utils import gen_transition_gt
except:
    import transforms as T
    from utils import gen_transition_gt

class ImageMatteDataset(Dataset):
    def __init__(self, root_dir, split, short_size=1024, is_train=False, crop=[512, 512], flip_p=0.5, random_seed=2023, **kwargs):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.is_train = is_train
        self.load_image_alphas()

        # Augementation
        self.random = np.random.RandomState(random_seed)

        self.transforms = [T.Load(), T.MasksFromBinarizedAlpha(), T.ResizeShort(short_size, transform_alphas=is_train), T.PaddingMultiplyBy(32, transform_alphas=is_train), T.Stack()]
        if self.is_train:
            self.transforms.extend([
                T.RandomCropByAlpha(crop, self.random),
                T.RandomHorizontalFlip(self.random, flip_p),
            ])
        self.transforms += [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transforms = T.Compose(self.transforms)
    
    def load_image_alphas(self):
        images = sorted(glob.glob(os.path.join(self.root_dir, "images", "*.jpg")))
        alphas = sorted(glob.glob(os.path.join(self.root_dir, "alphas", "*.png")))
        assert len(images) > 0, "No images found in {}".format(os.path.join(self.root_dir, "images"))
        assert len(images) == len(alphas), "Number of images and alphas are not matched"
        assert all([os.path.basename(image)[:-4] == os.path.basename(alpha)[:-4] for image, alpha in zip(images, alphas)]), "Image and alpha names are not matched"
        self.data = list(zip(images, alphas))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, alpha_path = self.data[idx]
        input_dict = {
            "frames": [image_path],
            "alphas": [alpha_path],
            "masks": None
        }
        output_dict = self.transforms(input_dict)
        image, alpha, mask, transform_info = output_dict["frames"], output_dict["alphas"], output_dict["masks"], output_dict["transform_info"]
        
        mask = F.interpolate(mask, size=(mask.shape[2] // 8, mask.shape[3] // 8), mode="nearest")

        # TODO: Build transition GT
        transition_gt = None
        if self.is_train:
            k_size = self.random.choice(range(2, 5))
            iterations = np.random.randint(5, 15)
            transition_gt = gen_transition_gt(alpha, mask, k_size, iterations)

        alpha = alpha * 1.0 / 255
        mask = mask * 1.0 / 255

        if mask.sum() == 0:
            # logging.error("Get another sample, alphas are incorrect: {}".format(alpha_path))
            return self.__getitem__(self.random.randint(0, len(self.data)))

        out =  {'image': image, 'mask': mask.float(), 'alpha': alpha.float()}
        out['fg'] = output_dict.get('fg', image)
        out['bg'] = output_dict.get('bg', image)
        
        if not self.is_train:
            # Generate trimap for evaluation
            trans = gen_transition_gt(alpha)
            trimap = torch.zeros_like(alpha)
            trimap[alpha > 0.5] = 2.0 # FG
            trimap[trans > 0] = 1.0 # Transition
            out.update({'trimap': trimap, 'image_names': [image_path], 'transform_info': transform_info, "skip": 0})
        else:
            out.update({'transition': transition_gt.float()})
        
        return out

if __name__ == "__main__":
    import cv2
    train_dataset = ImageMatteDataset("/mnt/localssd/HHM", split="train", is_train=True)
    val_dataset = ImageMatteDataset("/mnt/localssd/HHM", split="val", is_train=False)

    for batch in val_dataset:
        image = batch["image"][0]
        mask = batch["mask"][0]
        alpha = batch["alpha"][0]
        # transition = batch["transition"][0]
        trimap = batch["trimap"][0]
        idx = 0
        frame = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        frame = (frame * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        cv2.imwrite("frame_{}.png".format(idx), frame[:, :, ::-1])
        cv2.imwrite("mask_{}.png".format(idx), mask[0].numpy() * 255)
        cv2.imwrite("alpha_{}.png".format(idx), alpha[0].numpy() * 255)
        # cv2.imwrite("transition_{}.png".format(idx), transition[0].numpy() * 255)
        cv2.imwrite("trimap_{}.png".format(idx), trimap[0].numpy() * 80)
        print(batch["image_names"])
        print(batch["transform_info"])
        import pdb; pdb.set_trace()