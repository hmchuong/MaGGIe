import os
import glob
import shutil
import numpy as np
import cv2
from scipy.ndimage import label as scipy_label

input_dir = "/home/chuongh/vm2m/data/vis/real_qual"
output_dir = "/home/chuongh/vm2m/data/vis/real_qual_extracted_1126"
target_dirs = [
    "masks",
    "instmatt",
    "sparsemat_hr",
    "mgm_single_tcvom",
    "mgm_stacked_tcvom",
    "ours"
]
video_info = [
    ["3_pexels-cottonbro-5329478_2160p", "00", 165, 3],
    ["3_production_id_4122569_2160p", "00", 130, 3],
    ["3_production_id_4122569_2160p", "01", 35, 3],
    ["6_production_id_4880458_2160p", "02", 555, 3],
    ["3_production_id_4122569_2160p", "02", 115, 3],
    ["6_production_id_4880458_2160p", "01", 365, 3]
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

def postprocess(mask):
    regions, n_regions = scipy_label(mask > 0.1)
    if n_regions > 1:
        max_area = 0
        all_areas = np.zeros((n_regions + 1,))
        for i in range(n_regions):
            all_areas[i+1] = (regions == (i+1)).sum()
            max_area = max(max_area, all_areas[i+1])
        
        if max_area == 0:
            return
        
        if max_area * 1.0 / mask.size < 0.01:
            mask[:] = 0
            return

        # Get location of areas
        max_area_id = all_areas.argmax()

        # Remove noise
        for i in range(n_regions):
            if i + 1 == max_area_id: continue
            object_mask = regions == (i+1)
            if all_areas[i+1] < max_area * 0.1:
                mask[object_mask] = 0

for video_name, instance, frame_id, num_frames in video_info:
    out_dirname = "{}_{}_{}".format(video_name, instance, frame_id)

    image_frames = sorted(os.listdir(os.path.join(input_dir, "images", video_name)))
    start_framename = "{:05d}".format(frame_id)
    start_idx = image_frames.index(start_framename + ".jpg")
    image_frames = image_frames[start_idx:start_idx + num_frames]
    # Copy image first
    for image_frame in image_frames:
        src_path = os.path.join(input_dir, "images", video_name, image_frame)
        tgt_path = os.path.join(output_dir, out_dirname, image_frame)
        os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
        shutil.copy2(src_path, tgt_path)

    # Copy others
    for target_dir in target_dirs:
        prev_pred = None
        for image_frame in image_frames:
            image_frame = image_frame.replace(".jpg", "")
            src_path = os.path.join(input_dir, target_dir, video_name, image_frame, "{}.png".format(instance))
            tgt_path = os.path.join(output_dir, out_dirname, "{}_{}_{}.png".format(image_frame, instance, target_dir))
            os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            shutil.copy2(src_path, tgt_path)

            if target_dir == "masks": continue
            curr_pred = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

            # compose with green background
            alpha = curr_pred / 255.0
            postprocess(alpha)
            cv2.imwrite(tgt_path, alpha * 255)
            alpha = alpha[:, :, None]
            alpha = np.repeat(alpha, 3, axis=2)

            image = cv2.imread(os.path.join(output_dir, out_dirname, image_frame + ".jpg"), cv2.IMREAD_COLOR)
            green_bg = np.zeros_like(image)
            green_bg[:, :, 1] = 255
            composed = (1 - alpha) * green_bg + alpha * image
            tgt_path = os.path.join(output_dir, out_dirname, "{}_{}_{}_fg.png".format(image_frame, instance, target_dir))
            cv2.imwrite(tgt_path, composed)
            if prev_pred is not None:
                diff = log_diff_cmap(alpha * 255, prev_pred)
                tgt_path = os.path.join(output_dir, out_dirname, "{}_{}_{}_diff.png".format(image_frame, instance, target_dir))
                cv2.imwrite(tgt_path, diff)
            prev_pred = alpha * 255