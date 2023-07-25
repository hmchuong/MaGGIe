import os
import glob
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from .utils import gen_transition_gt
except ImportError:
    from utils import gen_transition_gt

class ComposedInstImageMatteDataset(Dataset):
    def __init__(self, root_dir, split, bg_dir, max_inst=10, padding_inst=10, short_size=768, crop=(512, 512), random_seed=2023):
        super().__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.load_image_alphas()
        self.bg_images = self.load_bg(bg_dir)
        self.max_inst = max_inst
        self.padding_inst = padding_inst

        self.short_size = short_size
        self.crop = crop

        # Augmentation
        self.random = np.random.RandomState(random_seed)

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def load_image_alphas(self):
        images = sorted(glob.glob(os.path.join(self.root_dir, "images", "*.jpg")))
        alphas = sorted(glob.glob(os.path.join(self.root_dir, "alphas", "*.png")))
        assert len(images) > 0, "No images found in {}".format(os.path.join(self.root_dir, "images"))
        assert len(images) == len(alphas), "Number of images and alphas are not matched"
        assert all([os.path.basename(image)[:-4] == os.path.basename(alpha)[:-4] for image, alpha in zip(images, alphas)]), "Image and alpha names are not matched"
        self.data = list(zip(images, alphas))
    
    def load_bg(self, bg_dir):
        ''' Load background image paths
        '''
        bg_images = []
        for image_name in os.listdir(bg_dir):
            image_path = os.path.join(bg_dir, image_name)
            bg_images.append(image_path)
        return bg_images
    
    def __len__(self):
        return len(self.data)

    def prepare_fg(self, image, alpha, max_w, max_h):
        # Crop fg
        ys, xs = np.where(alpha > 10)
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        image = image[ymin:ymax+1, xmin:xmax+1]
        alpha = alpha[ymin:ymax+1, xmin:xmax+1]

        # Resize fg
        h, w = image.shape[:2]
        scale = self.random.rand() * 0.5 + 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        image = image[:max_h, :max_w]
        alpha = alpha[:max_h, :max_w]

        # Flip fg
        if self.random.choice([True, False]):
            image = image[:, ::-1]
            alpha = alpha[:, ::-1]

        # Blur fg
        if self.random.choice([True, False]):
            kernel_size = self.random.choice([5, 15, 25])
            sigma = self.random.choice([1.0, 1.5, 3.0, 5.0])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        return image, alpha / 255.0

    def binarized_alpha(self, alpha):
        '''
        alpha: (H, W)
        '''
        alpha = alpha * 255
        threshold = np.random.uniform(0.1, 0.95) * 255

        # binarize the alphas
        _, img_binarized = cv2.threshold(np.array(alpha), threshold, 1, cv2.THRESH_BINARY)
        
        # generate random kernel sizes for dilation and erosion
        kernel_size_dilate = self.random.randint(1, 30)
        kernel_size_erode = self.random.randint(1, 30)

        # create the kernels
        kernel_dilate = np.ones((kernel_size_dilate, kernel_size_dilate), np.uint8)
        kernel_erode = np.ones((kernel_size_erode, kernel_size_erode), np.uint8)

        # randomly decide the order of dilation and erosion and whether to do both or just one
        operation_order = self.random.choice(["dilate_erode", "erode_dilate", "dilate", "erode"])

        if operation_order == "dilate_erode":
            img_dilated = cv2.dilate(img_binarized, kernel_dilate, iterations=1)
            img_final = cv2.erode(img_dilated, kernel_erode, iterations=1)
        elif operation_order == "erode_dilate":
            img_eroded = cv2.erode(img_binarized, kernel_erode, iterations=1)
            img_final = cv2.dilate(img_eroded, kernel_dilate, iterations=1)
        elif operation_order == "dilate":
            img_final = cv2.dilate(img_binarized, kernel_dilate, iterations=1)
        else:  # operation_order == "erode"
            img_final = cv2.erode(img_binarized, kernel_erode, iterations=1)
        return img_final

    def __getitem__(self, index):
        
        # Choose no. of instances
        no_instances = self.random.choice(range(1, self.max_inst + 1))

        # Load images and alphas
        images = []
        alphas = []
        idx = self.random.choice(range(0, len(self.data)), size=no_instances, replace=False)
        data = [self.data[i] for i in idx]

        for image_path, alpha_path in data:
            image = Image.open(image_path).convert('RGB')
            alpha = Image.open(alpha_path).convert('L')
            images.append(np.array(image))
            alpha = np.array(alpha)
            if len(alpha.shape) == 3:
                alpha = alpha[:, :, 0]
            alphas.append(alpha)

        # Choose bg image: from the first image or from bg_dir
        using_bg = self.random.choice([True, False])
        bg_image = images[0]
        if using_bg:
            bg_image = self.random.choice(self.bg_images)
            bg_image = Image.open(bg_image).convert('RGB')
            bg_image = np.array(bg_image)

        if not using_bg:
            images[0] = bg_image
            alphas[0] = alphas[0] / 255.0

        # Augment: resize, flip, blur, etc.
        max_h, max_w = bg_image.shape[:2] # max width and height to resize
        for i in range(0 if using_bg else 1, no_instances):
            images[i], alphas[i] = self.prepare_fg(images[i], alphas[i], max_w, max_h)

        # Compose images and alphas
        composed_image = np.zeros_like(bg_image)
        composed_alpha = np.zeros((no_instances, max_h, max_w), dtype=np.float32)

        # import pdb; pdb.set_trace()
        for i in range(no_instances):
            if i == 0 and not using_bg:
                # Use the first image as bg
                composed_image[:] = images[i]
                composed_alpha[i] = alphas[i]
            else:
                # Find x, y location to place image
                h, w = images[i].shape[:2]
                if h == max_h:
                    y = 0
                else:
                    y = self.random.randint(0, max_h - h)
                if w == max_w:
                    x = 0
                else:
                    x = self.random.randint(0, max_w - w)

                # Place image on bg
                composed_image[y: y + h, x: x+w] = (composed_image[y: y + h, x: x + w] * (1 - alphas[i][..., None]) \
                                                    + images[i] * alphas[i][..., None]).astype('uint8')

                # Place alpha
                composed_alpha[i, y: y + h, x: x+w] = alphas[i]
                composed_alpha[:i, y: y + h, x: x+w] *= (1 - alphas[i][None])

        roi_region = composed_alpha.sum(axis=0) > 0.5
        if roi_region.sum() == 0:
            return self.__getitem__(0)
        
        # Crop images and alphas (random crop)
        ratio = self.short_size / min(bg_image.shape[:2])
        composed_image = cv2.resize(composed_image, (int(bg_image.shape[1] * ratio), int(bg_image.shape[0] * ratio)), interpolation=cv2.INTER_CUBIC)
        composed_alpha = np.stack([cv2.resize(a, (composed_image.shape[1], composed_image.shape[0]), interpolation=cv2.INTER_CUBIC) for a in composed_alpha], axis=0)

        x, y = self.random.randint(0, composed_image.shape[1] - self.crop[0]), self.random.randint(0, composed_image.shape[0] - self.crop[1])
        composed_image = composed_image[y: y + self.crop[1], x: x + self.crop[0]]
        composed_alpha = composed_alpha[:, y: y + self.crop[1], x: x + self.crop[0]]

        # Binarize alpha to have input masks
        masks = np.stack([self.binarized_alpha(alpha) for alpha in composed_alpha], axis=0)
        masks = masks.astype(np.float32) # (n_inst, H, W)
        masks = torch.from_numpy(masks)
        masks = masks.unsqueeze(0) # (1, n_inst, H, W)
        masks = F.interpolate(masks, scale_factor=1.0/8, mode='nearest') # (1, n_inst, H/8, W/8)
        
        # Prepare inputs
        # Normalize image
        composed_image = self.to_tensor(composed_image) # (3, H, W)

        # Padding masks and alphas
        composed_alpha = torch.from_numpy(composed_alpha).unsqueeze(0) # (1, n_inst, H, W)

        add_padding = self.padding_inst - no_instances
        if add_padding > 0:
            composed_alpha = torch.cat([composed_alpha, torch.zeros(1, add_padding, *composed_alpha.shape[2:])], dim=1)
            masks = torch.cat([masks, torch.zeros(1, add_padding, *masks.shape[2:])], dim=1)
       
        # Compute transition GT
        k_size = self.random.choice(range(2, 5))
        iterations = np.random.randint(5, 15)
        transition_gt = gen_transition_gt(composed_alpha[0, :, None], None, k_size, iterations)[:, 0]

        out = {
            "image": composed_image.unsqueeze(0), # (1, 3, H, W)
            "alpha": composed_alpha, # (1, n_inst, H, W)
            "mask": masks, # (1, n_inst, H/8, W/8)
            'transition': transition_gt.unsqueeze(0).float(), # (1, n_inst, H, W)
        }

        return out
    
if __name__ == "__main__":
    train_dataset = ComposedInstImageMatteDataset(root_dir='/mnt/localssd/HHM', split='train', bg_dir='/mnt/localssd/bg', max_inst=3, short_size=768, crop=(512, 512), random_seed=2023)
    # 
    for batch in train_dataset:
        frames, masks, alphas, transition_gt = batch["image"], batch["mask"], batch["alpha"], batch["transition"]
        frame = frames[0] * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        frame = (frame * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        cv2.imwrite("frame.png", frame[:, :, ::-1])
        for idx in range(masks.shape[1]):
            mask = masks[0, idx]
            alpha = alphas[0, idx]
            transition = transition_gt[0, idx]
            cv2.imwrite("mask_{}.png".format(idx), mask.numpy() * 255)
            cv2.imwrite("alpha_{}.png".format(idx), alpha.numpy() * 255)
            cv2.imwrite("transition_{}.png".format(idx), transition.numpy() * 255)
        import pdb; pdb.set_trace()