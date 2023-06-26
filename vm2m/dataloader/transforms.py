import numpy as np
import cv2
import torch
from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_dict: dict):
        transform_info = []
        input_dict["transform_info"] = transform_info
        for t in self.transforms:
            input_dict = t(input_dict)
            # import pdb; pdb.set_trace()
        return input_dict

class Load(object):
    def __init__(self, is_rgb=True):
        self.is_rgb = is_rgb
    def __call__(self, input_dict: dict):
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]
        frames = [np.array(Image.open(frame_path).convert("RGB" if self.is_rgb else "BGR")) for frame_path in frames]
        if masks is not None:
            masks = [np.array(Image.open(mask_path).convert("L")) for mask_path in masks]
        alphas = [np.array(Image.open(alpha_path).convert("L")) for alpha_path in alphas]
        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        input_dict["masks"] = masks
        return input_dict
        
class ResizeShort(object):
    def __init__(self, short_size, transform_alphas=True):
        self.short_size = short_size
        self.transform_alphas = transform_alphas
    
    def __call__(self, input_dict: dict):
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]
        transform_info = input_dict["transform_info"]
        h, w = frames[0].shape[:2]
        ratio = self.short_size * 1.0 / min(w, h) 
        if ratio != 1:
            frames = [cv2.resize(frame, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR) for frame in frames]
            if masks is not None:
                masks = [cv2.resize(mask, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST) for mask in masks]
            # import pdb; pdb.set_trace()
            if self.transform_alphas:
                alphas = [cv2.resize(alpha, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR) for alpha in alphas]
        transform_info.append({'name': 'resize', 'ori_size': (h, w), 'ratio': ratio})
        
        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        input_dict["masks"] = masks
        input_dict["transform_info"] = transform_info
        
        return input_dict

class PaddingMultiplyBy(object):
    def __init__(self, divisor=32, transform_alphas=True):
        self.divisor = divisor
        self.transform_alphas = transform_alphas

    def __call__(self, input_dict: dict):
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]
        transform_info = input_dict["transform_info"]

        h, w = frames[0].shape[:2]
        h_pad = (self.divisor - h % self.divisor) % self.divisor
        w_pad = (self.divisor - w % self.divisor) % self.divisor
        frames = [cv2.copyMakeBorder(frame, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0) for frame in frames]
        if masks is not None:
            masks = [cv2.copyMakeBorder(mask, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0) for mask in masks]
        if self.transform_alphas:
            alphas = [cv2.copyMakeBorder(alpha, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0) for alpha in alphas]
        transform_info.append({'name': 'padding', 'pad_size': (h_pad, w_pad)})
        
        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        input_dict["masks"] = masks
        input_dict["transform_info"] = transform_info

        return input_dict

class Stack(object):
    def __init__(self):
        pass
    def __call__(self, input_dict: dict):
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]

        frames = np.stack(frames, axis=0)
        alphas = np.stack(alphas, axis=0)
        if masks is not None:
            masks = np.stack(masks, axis=0)
        
        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        input_dict["masks"] = masks

        return input_dict

class RandomCropByAlpha(object):
    def __init__(self, crop_size, random):
        self.crop_size = crop_size
        self.random = random
    
    def __call__(self, input_dict: dict):
        '''
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) or None
        '''

        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]

        h, w = frames[0].shape[:2]
        if h < self.crop_size[0] or w < self.crop_size[1]:
            raise ValueError("Crop size {} is larger than image size {}".format(self.crop_size, (h, w)))
        
        # Find alpha region
        try:
            ys, xs = np.where(alphas.mean(0) > 127)
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
        except:
            min_x, max_x = 0, w
            min_y, max_y = 0, h


        max_x = max(max_x - self.crop_size[1], min_x + 1)
        max_y = max(max_y - self.crop_size[0], min_y + 1)

        for _ in range(3):
            x, y = self.random.randint(min_x, max_x), self.random.randint(min_y, max_y)

            x = min(x, w - self.crop_size[1])
            y = min(y, h - self.crop_size[0])

            crop_frames = frames[:, y:y+self.crop_size[0], x:x+self.crop_size[1], :]
            crop_alphas = alphas[:, y:y+self.crop_size[0], x:x+self.crop_size[1]]
            if (crop_alphas > 127).sum() > 0:
                break

        crop_masks = masks
        if masks is not None:
            crop_masks = masks[:,y:y+self.crop_size[0], x:x+self.crop_size[1]]
        
        input_dict["frames"] = crop_frames
        input_dict["alphas"] = crop_alphas
        input_dict["masks"] = crop_masks

        return input_dict

class RandomHorizontalFlip(object):
    def __init__(self, random, p=0.5):
        self.random = random
        self.p = p

    def __call__(self, input_dict: dict):
        '''
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) or None
        '''
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]

        if self.random.rand() < self.p:
            frames = frames[:, :, ::-1, :]
            alphas = alphas[:, :, ::-1]
            if masks is not None:
                masks = masks[:, :, ::-1]
        
        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        input_dict["masks"] = masks

        return input_dict

class RandomComposeBackground(object):
    def __init__(self, bg_paths, random, is_rgb=True, blur_p=0.5, blur_kernel_size=[5, 15, 25], blur_sigma=[1.0, 1.5, 3.0, 5.0]):
        self.bg_paths = bg_paths
        self.random = random
        self.is_rgb = is_rgb
        self.blur_p = blur_p
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
    
    def __call__(self, input_dict: dict):
        '''
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) or None
        '''
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]

        bg_path = self.random.choice(self.bg_paths)
        bg = cv2.imread(bg_path)
        if self.is_rgb:
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        
        # Blur background
        if self.random.rand() < self.blur_p:
            blur_ks = self.random.choice(self.blur_kernel_size)
            blur_sigma = self.random.choice(self.blur_sigma)
            bg = cv2.GaussianBlur(bg, (blur_ks, blur_ks), blur_sigma)
        
        # Crop background following frames size
        h, w = frames[0].shape[:2]
        bh, bw = bg.shape[:2]
        x = self.random.randint(0, bw - w)
        y = self.random.randint(0, bh - h)
        bg = bg[y:y+h, x:x+w, :]
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)

        # Compose background
        frames = frames.astype(np.float32)
        bg = bg.astype(np.float32)
        compose_frames = frames * (alphas[..., None].astype(float) / 255.0) + bg * (1 - alphas[..., None].astype(float) / 255.0)
        compose_frames = np.clip(compose_frames, 0, 255).astype(np.uint8)
        input_dict["fg"] = frames
        input_dict["bg"] = np.tile(bg, (frames.shape[0], 1, 1, 1))
        
        input_dict["frames"] = compose_frames
        
        return input_dict

class MasksFromBinarizedAlpha(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def __call__(self, input_dict: dict):
        
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]

        if masks is None:
            masks = [(a > self.threshold * 255).astype(np.uint8) * 255 for a in alphas]
        
        input_dict["masks"] = masks

        return input_dict
    
class RandomBinarizeAlpha(object):
    def __init__(self, random, binarize_max_k=30):
        self.random = random
        self.binaraize_max_k = binarize_max_k

    def _gen_single_mask(self, alpha):
        '''
        alpha: (H, W)
        '''
        
        threshold = np.random.uniform(0.1, 0.95) * 255

        # binarize the alphas
        _, img_binarized = cv2.threshold(np.array(alpha), threshold, 1, cv2.THRESH_BINARY)

        # generate random kernel sizes for dilation and erosion
        kernel_size_dilate = self.random.randint(1, self.binaraize_max_k)
        kernel_size_erode = self.random.randint(1, self.binaraize_max_k)

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
        return img_final * 255

    def __call__(self, input_dict: dict):
        '''
        Args:
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) or None
        
        Args:
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) from alphas
        '''
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]

        if masks is None:
            masks = np.stack([self._gen_single_mask(alpha) for alpha in alphas], axis=0)
        
        input_dict["masks"] = masks

        return input_dict
        

class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self, input_dict: dict):
        '''
        Args:
        frames: (T, H, W, C)
        alphas: (T, H, W)
        masks: (T, H, W) or None

        Returns:
        frames: (T, C, H, W)
        alphas: (T, 1, H, W)
        masks: (T, 1, H, W) or None
        '''
        frames = input_dict["frames"]
        alphas = input_dict["alphas"]
        masks = input_dict["masks"]

        frames = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).contiguous().float()        
        alphas = torch.from_numpy(np.ascontiguousarray(alphas)).unsqueeze(1).contiguous()

        if masks is not None:
            masks = torch.from_numpy(np.ascontiguousarray(masks).astype('uint8')).unsqueeze(1).contiguous()

        input_dict["frames"] = frames
        input_dict["alphas"] = alphas
        input_dict["masks"] = masks
        
        if "fg" in input_dict:
            input_dict["fg"] = torch.from_numpy(np.ascontiguousarray(input_dict["fg"])).permute(0, 3, 1, 2).contiguous().float()
        if "bg" in input_dict:
            input_dict["bg"] = torch.from_numpy(np.ascontiguousarray(input_dict["bg"])).permute(0, 3, 1, 2).contiguous().float()
        return input_dict

class Normalize(object):
    def __init__(self, mean=[], std=[]):
        self.mean = mean
        self.std = std
    
    def norm(self, frames):
        mean = torch.tensor(self.mean).view(1, 3, 1, 1).float()
        std = torch.tensor(self.std).view(1, 3, 1, 1).float()
        frames = frames / 255.0
        frames = (frames - mean) / std
        return frames
    
    def __call__(self, input_dict: dict):
        frames = input_dict["frames"]    
        input_dict["frames"] = self.norm(frames)
        if "fg" in input_dict:
            input_dict["fg"] = self.norm(input_dict["fg"])
        if "bg" in input_dict:
            input_dict["bg"] = self.norm(input_dict["bg"])
        return input_dict