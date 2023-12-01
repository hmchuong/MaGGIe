import glob
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

# Extract fg, bg based on frame and alpha

def process(args):
    fg_path, alpha_path = args
    _f = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
    _a = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
    _a[_a <= 1] = 0

    # Generate FG and BG by remove color by alpha
    _bg = _f.copy()
    _af = _a.astype(np.float32) / 255.
    _af = _af[..., np.newaxis]
    _af = np.repeat(_af, 3, axis=2)
    fg_region = (_f[_af > 0].astype(np.float32) / _af[_af > 0]).astype(np.uint8)
    fg_region = np.clip(fg_region, 0, 255)
    bg_region = (_f[_af > 0].astype(np.float32) - fg_region * _af[_af >0]) / (1. - _af[_af > 0])
    np.nan_to_num(bg_region, copy=False)
    bg_region = np.clip(bg_region, 0, 255)
    bg_region = bg_region.astype(np.uint8)
    _f[_af > 0] = fg_region
    _bg[_af > 0] = bg_region
    
    fgr = Image.fromarray(_f[:, :, ::-1])
    bgr = Image.fromarray(_bg[:, :, ::-1])
    
    out_fg_path = alpha_path.replace("/pha/", "/fgr_extracted/").replace(".png", ".jpg")
    out_bg_path = alpha_path.replace("/pha/", "/bgr_extracted/").replace(".png", ".jpg")
    os.makedirs(os.path.dirname(out_fg_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_bg_path), exist_ok=True)
    fgr.save(out_fg_path)
    bgr.save(out_bg_path)

if __name__ == "__main__":
    out_file = "/sensei-fs/users/chuongh/InstMatt/datasets/VIM_train"
    data_root = "/mnt/localssd/syn/train"    
    all_fgrs = []
    all_alphas = []

    video_names = glob.glob(os.path.join(data_root, "fgr") + "/*")
    video_names.sort()

    for video_name in video_names:
        framenames = glob.glob(os.path.join(video_name, "*.jpg"))
        framenames.sort()
        alpha_names = glob.glob(video_name.replace("fgr", "pha") + "/*/*.png")
        alpha_names.sort()
        n_i = len(alpha_names) // len(framenames)
        for j in range(n_i):
            all_fgrs.extend(framenames)
            all_alphas.extend([x.replace("fgr", "pha").replace(".jpg", "") + f"/{j:02d}.png" for x in framenames])

    params = list(zip(all_fgrs, all_alphas))
    
    with Pool(80) as p:
        pbar = tqdm(total=len(params))
        for i, _ in tqdm(enumerate(p.imap_unordered(process, params))):
            pbar.update()