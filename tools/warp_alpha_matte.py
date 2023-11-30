import os
import cv2
from PIL import Image
import torch
import numpy as np
from argparse import Namespace
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('RAFT/core')
from raft import RAFT
from utils.utils import InputPadder, coords_grid, bilinear_sampler
from multiprocessing import Pool

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    device = x.device
    grid = grid.to(device)
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to(device)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1
    
    return output, mask

def load_image(imfile, is_gray=False, device='cuda', resize=True):
    img = Image.open(imfile)
    if is_gray:
        img = img.convert('L')
    ratio = 1920.0 / max(img.size)
    if ratio < 1 and resize:
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.BILINEAR)
    img = np.array(img).astype(np.uint8)
    if len(img.shape) == 2:
        img = img[:, :, None]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def get_occlusion_mask(H, W, flow_right, flow_left, threshold=30):
    
    coords0 = coords_grid(1, H, W, flow_right.device)

    coords1 = coords0 + flow_left
    coords2 = coords1 + bilinear_sampler(flow_right, coords1.permute(0,2,3,1))

    err = (coords0 - coords2).norm(dim=1)

    mask = (err > threshold).float()
    # Image.fromarray((mask[0].cpu().numpy() * 255).astype('uint8')).save("test_mask.png")
    # import pdb; pdb.set_trace()
    return mask[:, None]

    # mesh grid
    # xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    # yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    # xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    # yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    # grid = torch.cat((xx, yy), 1).float().to(flow_right.device)

    # new_grid = grid + flow_right + flow_left
    
    # dist = torch.sqrt(torch.pow(new_grid - grid, 2).sum(1))

    # # crop = dist[:, 246:256, 855:865]
    # import pdb; pdb.set_trace()
    
    # dist = (dist > threshold).float()
    # Image.fromarray(((dist < 1)[0].cpu().numpy() * 255).astype('uint8')).save("test_mask.png")
    
    # return dist[:, None]
    

def process_videos(device, video_names):
    args = Namespace(small=False, mixed_precision=True, alternate_corr=False)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load('RAFT/models/raft-sintel.pth', map_location='cpu'))
    model = model.module
    model.to(device)
    _ = model.eval()

    img_dir = "/home/chuongh/vm2m/data/VIPSeg/out/fgr"
    pha_dir = "/home/chuongh/vm2m/data/VIPSeg/out/pha_vid_0911_from-seg"
    out_dir = "/home/chuongh/vm2m/data/VIPSeg/out/pha_vid_0911_from-seg_refined"
    
    eroded_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    for video_name in video_names:
        print("Processing", video_name)
        # Load frame pairs
        frame_names = sorted(os.listdir(os.path.join(img_dir, video_name)))

        # Count number of instances
        num_instances = len(os.listdir(os.path.join(pha_dir, video_name, frame_names[0].replace(".jpg", ""))))
        for inst_i in range(num_instances):
            pha_name = '%02d.png' % inst_i
            prev_pha = None

            # Copy the first frame
            output_path = os.path.join(out_dir, video_name, frame_names[0].replace(".jpg", ""), pha_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.open(os.path.join(pha_dir, video_name, frame_names[0].replace(".jpg", ""), pha_name)).save(output_path)

            for i in range(len(frame_names)-1):
                j = i + 1
                fgr_path_1 = os.path.join(img_dir, video_name, frame_names[i])
                fgr_path_2 = os.path.join(img_dir, video_name, frame_names[j])
                pha_path_1 = os.path.join(pha_dir, video_name, frame_names[i].replace(".jpg", ""), pha_name)
                pha_path_2 = os.path.join(pha_dir, video_name, frame_names[j].replace(".jpg", ""), pha_name)
                
                output_path = os.path.join(out_dir, video_name, frame_names[j].replace(".jpg", ""), pha_name)
                if os.path.exists(output_path):
                    continue

                # Ignore if either the source mask or target mask is zeros
                pha_1 = np.array(Image.open(pha_path_1).convert('L')) / 255.0
                pha_2 = np.array(Image.open(pha_path_2).convert('L')) / 255.0
                if (pha_1 < 0.5).sum() == 0 or (pha_2 < 0.5).sum() == 0:
                    prev_pha = None
                    continue
                
                # try:
                # Compute flows
                source_img = load_image(fgr_path_1, device=device, resize=True)
                target_img = load_image(fgr_path_2, device=device, resize=True)
                source_padder = InputPadder(source_img.shape)
                source_img, target_img = source_padder.pad(source_img, target_img)

                # import pdb; pdb.set_trace()
                with torch.no_grad():
                    _, flow_right = model(source_img, target_img, iters=20, test_mode=True)
                    _, flow_left = model(target_img, source_img, iters=20, test_mode=True)

                flow_right = source_padder.unpad(flow_right)
                flow_left = source_padder.unpad(flow_left)

                # Correct the pha_matte_2
                target_pha = load_image(pha_path_2, is_gray=True, device=device, resize=False)
                source_pha = load_image(pha_path_1, is_gray=True, device=device, resize=False) if prev_pha is None else prev_pha.to(device)

                # Resize the source_phamatte to the size of the target
                # source_pha = F.interpolate(source_pha, size=(target_pha.shape[2], target_pha.shape[3]), mode='bilinear')

                # Resize flow_left and flow_right to the size of the target
                flow_left = F.interpolate(flow_left, size=(target_pha.shape[2], target_pha.shape[3]), mode='bilinear')
                flow_right = F.interpolate(flow_right, size=(target_pha.shape[2], target_pha.shape[3]), mode='bilinear')
                occlusion_mask = get_occlusion_mask(target_pha.shape[-2], target_pha.shape[-1], flow_right, flow_left, threshold=1)


                # warped_source, _ = warp(target_pha / 255.0, flow_right)
                # warped_target, _ = warp(warped_source, flow_left)

                warped_target_2, mask_target = warp(source_pha / 255.0, flow_left)

                # Correct it
                # correct_mask = (1.0 - torch.abs(target_pha - warped_target * 255) / 255) * (mask_target == 1)
                # correct_mask = correct_mask.float()
                correct_mask = (1.0 - occlusion_mask) * (mask_target == 1)

                # Scale the mask back to the original size
                # correct_mask = F.interpolate(correct_mask, size=(pha_1.shape[0], pha_1.shape[1]), mode='nearest')
                # warped_target_2 = F.interpolate(warped_target_2, size=(pha_1.shape[0], pha_1.shape[1]), mode='bilinear')
                # target_pha = np.array(Image.open(pha_path_2).convert('L'))
                # target_pha = torch.from_numpy(target_pha)[None, None].to(device)

                new_target_pha = correct_mask.float() * (warped_target_2 * 255) + (1 - correct_mask.float()) * target_pha
                
                # Ignore occluded pixels
                # warped_target_2 = (warped_target_2 * 255) * correct_mask.float()

                # Fix foreground pixels in target_pha if there is a large discrepancy
                # fixing_mask = (target_pha > 30) & (torch.abs(target_pha - warped_target_2) > 10) & (warped_target_2 > 30)
                
                # Erode the warped_target_2 to avoid fixing the pixels at the boundary
                # fixing_mask = cv2.erode((warped_target_2[0,0].cpu().numpy() > 127).astype('uint8'), eroded_kernel, iterations=1)
                # fixing_mask = torch.from_numpy(fixing_mask).to(device)[None, None] > 0
                
                # warped_target_2 = F.interpolate(warped_target_2, size=(pha_1.shape[0], pha_1.shape[1]), mode='bilinear').type(torch.uint8)
                # fixing_mask = F.interpolate(fixing_mask.float(), size=(pha_1.shape[0], pha_1.shape[1]), mode='nearest').bool()
                # fixing_mask = fixing_mask #& (torch.abs(target_pha - warped_target_2) < 20)
                # target_pha[fixing_mask] = warped_target_2[fixing_mask]

                new_target_pha = new_target_pha.cpu()

                prev_pha = new_target_pha

                # Save the corrected pha_matte
                output_path = os.path.join(out_dir, video_name, frame_names[j].replace(".jpg", ""), pha_name)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                Image.fromarray(new_target_pha[0, 0].numpy().astype(np.uint8)).save(output_path)
                # except:
                #     print("Error in", video_name, frame_names[i], frame_names[j])
                    # import pdb; pdb.set_trace()
                    # continue
                # unknown_tensor = (new_target_pha > 1.0) & (new_target_pha < 255)
                # unknown_tensor = unknown_tensor.float()
                # Image.fromarray((unknown_tensor[0, 0].numpy() * 255).astype(np.uint8)).save("test_unknown.png")
                # Image.fromarray((correct_mask[0, 0].cpu().numpy() * 255).astype('uint8')).save("test_mask.png")
                # # Image.fromarray((fixing_mask[0, 0].cpu().numpy() * 255).astype('uint8')).save("test_mask.png")
                # Image.fromarray((warped_target_2[0, 0].cpu().numpy() * 255).astype('uint8')).save("test_warp.png")
                # Image.fromarray(target_pha[0, 0].cpu().numpy().astype(np.uint8)).save("test_target.png")
                # import pdb; pdb.set_trace()


if __name__ == "__main__":

    # Load number of videos
    video_names = sorted(os.listdir("/home/chuongh/vm2m/data/VIPSeg/out/mask"))
    np.random.shuffle(video_names)

    # Divide videos into subprocesses
    n_processes = 4
    video_names = np.array_split(video_names, n_processes)

    pool = Pool(n_processes).starmap(process_videos, [("cuda:0", video_names[0]), ("cuda:1", video_names[1]), ("cuda:2", video_names[2]), ("cuda:3", video_names[3])])
    pool.close()

    # Run subprocesses
    # process_videos("cuda:0", ["5_Yb4AMwr0vNE"])
