import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("video_name", type=str)
parser.add_argument("--instance", type=str, default="0")
argparse_args = parser.parse_args()
video_name = argparse_args.video_name

root_dir = "output/YTVOS"
instance_dir = "instance_{}".format(argparse_args.instance)
dirs = [
    f"/mnt/localssd/YTVOS/test/{video_name}",
    f"{root_dir}/mgm/test/mask/{video_name}/{instance_dir}",
    f"{root_dir}/mgm/test/alpha_pred/{video_name}/{instance_dir}",
    f"{root_dir}/tcvom/test/alpha_pred/{video_name}/{instance_dir}",
    f"{root_dir}/sparsemat/test/alpha_pred/{video_name}/{instance_dir}",
    f"{root_dir}/vm2m/test/alpha_pred/{video_name}/{instance_dir}",
    f"{root_dir}/vm2m/test/alpha_pred/{video_name}/{instance_dir}",
]
desc = [
    "image",
    "input mask",
    "MGM",
    "TCVOM",
    "SparseMat",
    "VM2M",
    "VM2M_0711",
]
dst_dir = f"{root_dir}/{video_name}_inst{argparse_args.instance}"
n_rows = 4
n_cols = 3

# create destination directory if it doesn't exist
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# get list of image filenames in the first directory
image_filenames = glob.glob(os.path.join(dirs[0], "*.jpg"))  # assuming the images are in png format

for image_filename in tqdm(image_filenames):
    images = []
    ori_image = None
    # try:
    for text, dir in zip(desc, dirs):
        if text is None:
            dummy = np.zeros_like(images[0])
            images.append(dummy)
            continue
        image_path = os.path.join(dir, os.path.basename(image_filename))
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
       
        if "/alpha_pred/" in dir:
            composed_img = np.zeros_like(image)
            composed_img[:, :, 1] = 255
            alpha = image * 1.0 / 255.0
            composed_img = (1.0 - alpha) * composed_img + alpha * ori_image
            composed_img = composed_img.astype(np.uint8)
            cv2.putText(composed_img, text, (6, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
            images.append(composed_img)
        if text == 'image':
            ori_image = image.copy()
        cv2.putText(image, text, (6, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
        images.append(image)

    # import pdb; pdb.set_trace()
    images = np.stack(images, axis=0)
    images = images.reshape(n_cols, n_rows, *images.shape[1:])
    images = np.transpose(images, (1, 0, 2, 3, 4))
    final_image = np.concatenate(images, axis=1)
    final_image = np.concatenate(final_image, axis=1)

    
    # save the final image in the destination directory
    max_w = 1920
    h, w = final_image.shape[:2]
    if w > max_w:
        final_image = cv2.resize(final_image, (max_w, int(h * max_w / w)))
    cv2.imwrite(os.path.join(dst_dir, os.path.basename(image_filename)), final_image)
    # except:
    #     pass
    # import pdb; pdb.set_trace()