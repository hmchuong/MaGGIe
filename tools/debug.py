import os

import cv2
import numpy as np
import torch
from PIL import Image

from vm2m.network.backbone import res_encoder_29
from vm2m.network.vm2m import VM2M

# Define model
backbone = res_encoder_29(late_downsample=True)
model = VM2M(backbone)
model = model.cuda()

# Load video (5 frames and 2 videos)
data_root = "/home/chuongh/mask2matte/data/VideoMatte240K/valid"
image_paths = f"{data_root}/comp"
mask_paths = f"{data_root}/coarse"
alpha_paths = f"{data_root}/pha"
video_name1 = "vm_0000"
video_name2 = "vm_0100"

def generator_tensor_dict(image_path, mask_path, alpha_path):
    # read images
    image = cv2.imread(image_path)

    max_size = 1024
    ratio = 1
    if max(image.shape) > max_size:
        ratio = max_size / max(image.shape)
        image = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
    
    # read masks
    mask = np.asarray(Image.open(mask_path).convert('P'))
    if ratio < 1:
        mask = cv2.resize(mask, (int(mask.shape[1] * ratio), int(mask.shape[0] * ratio)), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)

    # read alpha
    alpha = np.asarray(Image.open(alpha_path).convert('P'))
    if ratio < 1:
        alpha = cv2.resize(alpha, (int(alpha.shape[1] * ratio), int(alpha.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR)
    alpha = alpha / 255.0

    h, w = image.shape[:2]

    # Padding to divisible by 32
    target_h = 32 * ((h - 1) // 32 + 1)
    target_w = 32 * ((w - 1) // 32 + 1)
    pad_h = target_h - h
    pad_w = target_w - w
    padded_image = np.pad(image, ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
    padded_mask = np.pad(mask, ((0,pad_h), (0, pad_w)), mode="reflect")
    padded_alpha = np.pad(alpha, ((0,pad_h), (0, pad_w)), mode="reflect")

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    # swap color axis
    padded_image = padded_image.transpose((2, 0, 1)).astype(np.float32)

    # normalize image
    padded_image /= 255.

    # normalize image
    padded_image, padded_mask, padded_alpha = torch.from_numpy(padded_image), torch.from_numpy(padded_mask), torch.from_numpy(padded_alpha)
    padded_image = padded_image.sub_(mean).div_(std)

    # downsample the mask
    padded_mask = padded_mask[::8, ::8]

    return padded_image, padded_mask, padded_alpha

def load_video(video_name):

    # Load masks (5 frames and 2 videos)
    images = []
    masks = []
    alphas = []

    filenames = sorted(os.listdir(os.path.join(image_paths, video_name)))

    for filename in filenames[:5]:
        image_path = os.path.join(image_paths, video_name, filename)
        mask_path = os.path.join(mask_paths, video_name, filename)
        alpha_path = os.path.join(alpha_paths, video_name, filename)

        image, mask, alpha = generator_tensor_dict(image_path, mask_path, alpha_path)
        images.append(image)
        masks.append(mask)
        alphas.append(alpha)

    images = torch.stack(images).cuda()
    masks = torch.stack(masks).cuda()
    alphas = torch.stack(alphas).cuda()

    return images, masks, alphas

images1, masks1, alphas1 = load_video(video_name1)
images2, masks2, alphas2 = load_video(video_name2)

images = torch.stack([images1, images2], dim=0)
masks = torch.stack([masks1, masks2], dim=0)[:, :, None, :, :]
alphas = torch.stack([alphas1, alphas2], dim=0)[:, :, None, :, :]


# forward
model(images, masks)