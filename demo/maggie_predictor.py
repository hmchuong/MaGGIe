# Load model directly
import sys
sys.path.append("../")

import numpy as np
from PIL import Image
import torch
from maggie.network.arch import MaGGIe
from maggie.dataloader import transforms as T
from maggie.utils.postprocessing import reverse_transform_tensor

# Create image model
image_model = MaGGIe.from_pretrained("chuonghm/maggie-image-him50k-cvpr24")
image_model = image_model.eval()
image_model = image_model.cuda()

# video_model = MaGGIe.from_pretrained("chuonghm/maggie-video-him50k-cvpr24")

frame_transforms = T.Compose([
    T.ResizeShort(576, transform_alphas=False),
    T.PaddingMultiplyBy(64, transform_alphas=False),
    T.Stack(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame, bin_masks):
    instance_ids = np.unique(bin_masks)
    instance_ids = instance_ids[instance_ids != 0]
    instance_masks = []
    for instance_id in instance_ids:
        instance_mask = (bin_masks == instance_id) * 255
        instance_masks.append(instance_mask.astype(np.uint8))
    input_dict = {
        "frames": [np.array(frame.convert("RGB"))],
        "alphas": instance_masks,
        "masks": instance_masks,
    }
    output_dict = frame_transforms(input_dict)
    return {
        "image": output_dict["frames"][None].cuda(),
        "mask": (output_dict["masks"] / 255.0)[None].cuda().float(),
    }, output_dict["transform_info"]

def predict_image_alpha_matte(input_image, masks):

    # Preprocess image
    batch, transform_info = preprocess_frame(input_image, masks)

    # Predict alpha matte
    with torch.no_grad():
        output = image_model(batch)
    
    # Postprocess alpha matte
    alpha = output['refined_masks']
    alpha = reverse_transform_tensor(alpha, transform_info).cpu().numpy()
    alpha[alpha <= 1.0/255.0] = 0.0
    alpha[alpha >= 254.0/255.0] = 1.0
    
    alpha = alpha[0, 0]
    image = np.array(input_image)
    green_bg = np.zeros_like(image)
    green_bg[:, :, 1] = 255
    output = []
    for i in range(len(alpha)):
        a = alpha[i][:, :, None]
        result = (image * a + (1 - a) * green_bg).astype(np.uint8)
        result = Image.fromarray(result)
        output.append(result)

    return output