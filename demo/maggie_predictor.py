# Load model directly
import sys
sys.path.append("../")
import os
import gc
import numpy as np
from PIL import Image
import torch
from maggie.network.arch import MaGGIe, MaGGIe_Temp
from maggie.dataloader import transforms as T
from maggie.dataloader import build_dataset
from maggie.utils.postprocessing import reverse_transform_tensor, postprocess
from maggie.utils import CONFIG

# Create image model
image_model = MaGGIe.from_pretrained("chuonghm/maggie-image-him50k-cvpr24")
image_model = image_model.eval()
image_model = image_model.cuda()

# Create video model
video_model = MaGGIe_Temp.from_pretrained("chuonghm/maggie-video-vim2k5-cvpr24")
video_model = video_model.eval()
video_model = video_model.cuda()
CONFIG.merge_from_file("demo_video.yaml")

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

def predict_video_alpha_matte(progress, start_p, end_p):
    
    dataset = build_dataset(CONFIG.dataset.test, is_train=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device("cuda")
    with torch.no_grad():

        all_preds = []
        all_image_names = []
        mem_feats = None
        prev_pred = None

        for idx, batch in enumerate(dataloader):

            image_names = batch.pop('image_names')
            if 'alpha_names' in batch:
                _ = batch.pop('alpha_names')
            
            transform_info = batch.pop('transform_info')
            _ = batch.pop('trimap').numpy()
            _ = batch.pop('alpha').numpy()

            is_first = batch.pop('is_first')[0]
            is_last = batch.pop('is_last')[0]
            
            if is_first:
                # Free the saving frames
                all_preds = []
                all_image_names = []
                mem_feats = None
                prev_pred = None
                torch.cuda.empty_cache()
                gc.collect()

            batch = {k: v.to(device) for k, v in batch.items()}

            if batch['mask'].sum() == 0:
                continue
            output = video_model(batch, mem_feat=mem_feats, prev_pred=prev_pred)
                
            alpha = output['refined_masks']
            prev_pred = alpha[:, 1].cpu()

            alpha = reverse_transform_tensor(alpha, transform_info).cpu().numpy()

            # Threshold some high-low values
            alpha[alpha <= 1.0/255.0] = 0.0
            alpha[alpha >= 254.0/255.0] = 1.0

            alpha = postprocess(alpha)

            # Fuse results
            # Store all results (3 frames) for the first batch
            if is_first:
                all_preds = alpha[0]
                all_image_names = image_names
            else:
                all_image_names += image_names[2:]
                
                # Remove t+1 in previous preds, adding t and t+1 in new preds
                all_preds = np.concatenate([all_preds[:-1], alpha[0, 1:]], axis=0)

            # Add features t-1 to mem_feat
            if mem_feats is None and 'mem_feat' in output:
                if isinstance(output['mem_feat'], tuple):
                    mem_feats = tuple(x[:, 0] for x in output['mem_feat'])
            

            # Save the first frame, overwrite the previous pred
            end_idx = 1 if not is_last else len(all_preds)
            for image_name, pred in zip(all_image_names[:end_idx], all_preds[:end_idx]):
                for i in range(len(pred)):
                    image = Image.open(image_name[0])
                    green_bg = np.zeros_like(np.array(image))
                    green_bg[:, :, 1] = 255
                    result = (np.array(image) * pred[i, :, :, None] + (1 - pred[i, :, :, None]) * green_bg).astype(np.uint8)
                    result = Image.fromarray(result)
                    suffix = os.path.basename(image_name[0]).replace(".jpg", "")
                    prefix = f"{i:02d}"
                    out_path = os.path.join("video_results/matte", f"{prefix}_{suffix}.jpg")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    result.save(out_path)

            # Remove the very first stored values
            if len(all_preds) > 3:
                all_preds = all_preds[-3:]
                all_image_names = all_image_names[-3:]
            
            progress((idx + 1) / len(dataloader) * (end_p - start_p) + start_p, "Running MaGGIe...")