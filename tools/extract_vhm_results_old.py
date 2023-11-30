import os
import glob
import shutil
import numpy as np
import cv2

input_dir = "/home/chuongh/vm2m/data/vis/teaser_2"
output_dir = "/home/chuongh/vm2m/data/vis/real_qual_extracted"
input_dir = "/home/chuongh/vm2m/output/VHM/ours_vhm_bi-temp_1108_2/vis_fusion_13k_removenoise"
output_dir = "/home/chuongh/vm2m/data/vis/medium_qual_extracted"
target_dirs = [
    "masks_medium",
    "alphas_medium",
    "comp_medium_none",
    "comp_medium_mono",
    "comp_medium_bi",
    "comp_medium_bi_fusion_mono",
    "comp_medium",
]
video_info = [
    # ["2_pexels-artem-podrez-6003997_2160p", "00", 60, 3], # [video_name, instance_id, start_frame]
    # ["2_pexels-artem-podrez-6003997_2160p", "00", 140, 3],
    # ["2_pexels-artem-podrez-6003997_2160p", "00", 160, 3],
    # ["2_pexels-artem-podrez-6003997_2160p", "01", 65, 3],
    # ["2_pexels-cottonbro-5329614_2160p", "00", 30, 3],
    # ["2_pexels-cottonbro-5329614_2160p", "00", 135, 3],
    # ("01", "00", 5, 2),
    # ("01", "01", 5, 2)
    # ["2_pexels-cottonbro-5329614_2160p", "00", 590, 3],
    # ["3_pexels-cottonbro-5329478_2160p", "00", 165, 3],
    # ["3_production_id_4122569_2160p", "01", 10, 3],
    # ["4_pexels-pavel-danilyuk-8058107_Original", "02", 150, 6],
    # ["4_pexels-pavel-danilyuk-8058107_Original", "00", 150, 6],
    # ["4_pexels-pavel-danilyuk-8058107_Original", "00", 25, 3],
    # ["4_pexels-pavel-danilyuk-8058107_Original", "02", 25, 3]
    ("00029", 0, 12, 4),
    ("00080", 0, 11, 2),
    ("00112", 0, 18, 2),
    ("00221", 0, 20, 6),
]

def log_diff_cmap(pred, gt):
    pred = pred[:, :, 0]
    gt = gt[:, :, 0]
    diff = np.abs(pred - gt)
    diff = np.log(diff + 1e-6)
    diff = diff / np.log(255)
    diff = (diff * 255).astype('uint8')
    diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    return diff


for video_name, instance, frame_id, num_frames in video_info:
    out_dirname = "{}_{}_{}".format(video_name, instance, frame_id)

    image_frames = sorted(os.listdir(os.path.join(input_dir, "images_medium", video_name)))
    start_framename = "{:05d}".format(frame_id)
    start_idx = image_frames.index(start_framename + ".jpg")
    image_frames = image_frames[start_idx:start_idx + num_frames]
    # Copy image first
    for image_frame in image_frames:
        src_path = os.path.join(input_dir, "images_medium", video_name, image_frame)
        tgt_path = os.path.join(output_dir, out_dirname, image_frame)
        os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
        shutil.copy2(src_path, tgt_path)

    # Copy others
    all_gts = {}
    for target_dir in target_dirs:
        prev_pred = None
        for image_frame in image_frames:
            image_frame = image_frame.replace(".jpg", "")
            src_path = os.path.join(input_dir, target_dir, video_name, image_frame, "{:02d}.png".format(instance))
            tgt_path = os.path.join(output_dir, out_dirname, "{}_{}_{}.png".format(image_frame, instance, target_dir))
            os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            shutil.copy2(src_path, tgt_path)

            if target_dir == "masks_medium": continue
            curr_pred = cv2.imread(src_path, cv2.IMREAD_COLOR)
            if target_dir == "alphas_medium":
                all_gts[image_frame] = curr_pred
            elif target_dir != "masks_medium":
                print(target_dir)
                diff = log_diff_cmap(curr_pred, all_gts[image_frame])
                tgt_path = os.path.join(output_dir, out_dirname, "{}_{}_{}_diff_gt.png".format(image_frame, instance, target_dir))
                cv2.imwrite(tgt_path, diff)

            # compose with green background
            alpha = curr_pred / 255.0
            image = cv2.imread(os.path.join(output_dir, out_dirname, image_frame + ".jpg"), cv2.IMREAD_COLOR)
            green_bg = np.zeros_like(image)
            green_bg[:, :, 1] = 255
            composed = (1 - alpha) * green_bg + alpha * image
            tgt_path = os.path.join(output_dir, out_dirname, "{}_{}_{}_fg.png".format(image_frame, instance, target_dir))
            cv2.imwrite(tgt_path, composed)
            if prev_pred is not None:
                diff = log_diff_cmap(curr_pred, prev_pred)
                tgt_path = os.path.join(output_dir, out_dirname, "{}_{}_{}_diff_temp.png".format(image_frame, instance, target_dir))
                cv2.imwrite(tgt_path, diff)
            prev_pred = curr_pred
                