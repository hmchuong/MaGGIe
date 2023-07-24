import glob
import cv2
import os
import tqdm
from multiprocessing import Pool

def process(image_path):
    mask_path = image_path.replace('images', 'alphas').replace('.jpg', '.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype('float32') / 255.
    if mask.sum() == 0:
        return image_path, mask_path
    return None, None

# Clean HHM50K
if __name__ == '__main__':
    root_dir = '/mnt/localssd/HHM/train'
    num_empty = 0
    image_paths = glob.glob(root_dir + '/images/*.jpg')
    with Pool(32) as p:
        with tqdm.tqdm(total=len(image_paths)) as pbar:
            for i, (image_path, mask_path) in enumerate(p.imap_unordered(process, image_paths)):
                if image_path:
                    num_empty += 1
                    os.remove(image_path)
                    os.remove(mask_path)
                pbar.update()
                pbar.desc = f'num_empty: {num_empty}'