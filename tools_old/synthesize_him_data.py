import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

global valid_fg, bg_paths, output_dir, image_output_dir, alpha_output_dir
valid_fg = []
bg_paths = []
output_dir = '/mnt/localssd/HHM/synthesized_2'
image_output_dir = os.path.join(output_dir, 'images')
alpha_output_dir = os.path.join(output_dir, 'alphas')
bg_output_dir = os.path.join(output_dir, 'bg')
fg_output_dir = os.path.join(output_dir, 'fg')

# random = np.random #np.random.RandomState(2023)

def generate_image(sample_id):
    
    random = np.random.RandomState(sample_id)

    # 1. Random 2-5 fg images
    fg_images = random.choice(valid_fg, size=(random.randint(2, 5),), replace=False)
    alpha_images = [fg_image.replace('images', 'alphas').replace('.jpg', '.png') for fg_image in fg_images]

    # Read images
    fg_images = [Image.open(fg).convert('RGB') for fg in fg_images]
    alpha_images = [Image.open(alpha).convert('L') for alpha in alpha_images]

    # 1a. Crop fg images and alpha images following the fg region
    for i in range(len(fg_images)):
        alpha_mask = np.array(alpha_images[i])
        alpha_mask = np.where(alpha_mask > 0, 1, 0).astype(np.uint8)
        coords = cv2.findNonZero(alpha_mask)
        x, y, w, h = cv2.boundingRect(coords)
        fg_images[i] = fg_images[i].crop((x, y, x + w, y + h))
        alpha_images[i] = alpha_images[i].crop((x, y, x + w, y + h))

    # 2. Random 1 bg
    bg_image = random.choice(bg_paths)
    bg_image = Image.open(bg_image)
    ori_bg_image = bg_image.copy()

    target_w, target_h = bg_image.size

    # 3. Random resize fg and alpha images such that the fg height is from 50% to 90% of the target_h
    for i in range(len(fg_images)):
        scale = random.uniform(0.5, 0.9) * target_h / fg_images[i].height
        fg_images[i] = fg_images[i].resize((int(fg_images[i].width * scale), int(fg_images[i].height * scale)))
        alpha_images[i] = alpha_images[i].resize((int(alpha_images[i].width * scale), int(alpha_images[i].height * scale)))

    # 4. Random paste fg images to bg image
    final_alpha = np.zeros((len(fg_images), target_h, target_w), dtype=np.float32)
    all_fgs = []
    for i in range(len(fg_images)):
        is_success = False
        new_alphas = final_alpha
        for _ in range(3):
            try:
                x = random.randint(0, target_w - fg_images[i].width)
                y = random.randint(0, target_h - fg_images[i].height)
            except:
                break

            # Compute new alpha mask
            new_alphas = final_alpha.copy()
            new_alphas[i, y:y + fg_images[i].height, x:x + fg_images[i].width] = np.array(alpha_images[i]) / 255.0
            for j in range(i):
                new_alphas[j] *= (1 - new_alphas[i])
            new_areas = new_alphas.sum((1, 2))
            old_areas = final_alpha.sum((1, 2))
            ratio = (new_areas / (old_areas + 1e-7))
            if np.any((old_areas > 0) & (ratio < 0.7)):
                # print(old_areas)
                # print(new_areas)
                continue
            is_success = True
            break
        
        if not is_success:
            all_fgs.append(None)
            continue
        bg_image.paste(fg_images[i], (x, y), alpha_images[i])
        empty_image = Image.new('RGB', (target_w, target_h), (0, 0, 0))
        empty_image.paste(fg_images[i], (x, y))
        all_fgs.append(empty_image)
        final_alpha = new_alphas

    if final_alpha.sum() == 0:
        return
    bg_image.save(os.path.join(image_output_dir, f'{sample_id}.jpg'))
    ori_bg_image.save(os.path.join(bg_output_dir, f'{sample_id}.jpg'))

    alpha_index = 0
    for j in range(len(final_alpha)):
        alpha = final_alpha[j]
        if sum(alpha.flatten()) == 0:
            continue
        alpha = Image.fromarray((alpha * 255).astype(np.uint8))
        output_alpha_path = os.path.join(alpha_output_dir, str(sample_id), f'{alpha_index}.png')
        os.makedirs(os.path.dirname(output_alpha_path), exist_ok=True)

        alpha.save(output_alpha_path)

        fg_path = os.path.join(fg_output_dir, str(sample_id), f'{alpha_index}.jpg')
        os.makedirs(os.path.dirname(fg_path), exist_ok=True)
        all_fgs[j].save(fg_path)

        alpha_index += 1
    # return sample_id, bg_image, final_alpha

if __name__ == "__main__":

    # Load all image paths
    image_root = '/mnt/localssd/HHM/train/images'
    invalid_names = set()
    with open('../vm2m/dataloader/invalid_him.txt', 'r') as f:
        for line in f.readlines():
            invalid_names.add(line.strip())
    for image_name in os.listdir(image_root):
        if image_name in invalid_names:
            continue
        image_path = os.path.join(image_root, image_name)
        valid_fg.append(image_path)
    
    # Load all background paths
    bg_root = '/mnt/localssd/bg/'
    for f in os.listdir(bg_root):
        bg_paths.append(os.path.join(bg_root, f))

    
    os.makedirs(output_dir, exist_ok=True)

    
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(alpha_output_dir, exist_ok=True)
    os.makedirs(bg_output_dir, exist_ok=True)

    # Generate images
    num_images = 50000
    with Pool(80) as p:
        pbar = tqdm(total=num_images)
        for _ in p.imap_unordered(generate_image, range(num_images)):
            
            pbar.update(1)