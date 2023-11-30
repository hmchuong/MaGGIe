import os
import glob
import shutil

root_dir = "/home/chuongh/vm2m/data/HIM2K"
subset = "combine"

all_mask_dir = glob.glob(root_dir + "/masks_matched_*")

# import pdb; pdb.set_trace()
for mask_dir in all_mask_dir:
    postfix = os.path.basename(mask_dir).replace("masks_matched_", "")
    # copy images
    img_paths = glob.glob(root_dir + "/images/natural/*.jpg")
    for img_path in img_paths:
        img_name = os.path.basename(img_path).replace(".jpg", "_{}.jpg".format(postfix))
        new_img_path = os.path.join(root_dir, "images", subset, img_name)
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
        shutil.copy(img_path, new_img_path)
    
    # copy masks
    image_dirnames = os.listdir(mask_dir + "/natural")
    for image_dir in image_dirnames:
        mask_paths = glob.glob(os.path.join(mask_dir, "natural", image_dir, "*.png"))
        for mask_path in mask_paths:
            mask_name = os.path.basename(mask_path)
            new_mask_path = os.path.join(root_dir, "masks", subset, image_dir + "_" + postfix, mask_name)
            os.makedirs(os.path.dirname(new_mask_path), exist_ok=True)
            shutil.copy(mask_path, new_mask_path)

            alpha_path = mask_path.replace("masks_matched" + postfix, "alphas")
            new_alpha_path = new_mask_path.replace("masks", "alphas")
            os.makedirs(os.path.dirname(new_alpha_path), exist_ok=True)
            shutil.copy(alpha_path, new_alpha_path)