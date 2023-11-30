import os
import shutil
image_root = "/mnt/localssd/HIM2K/images/natural_wo_gt"
pred_dirs = [
     "/mnt/localssd/HIM2K/alphas/natural",
     "/home/chuongh/vm2m/data/HIM2K/masks_matched_r50_fpn_3x/natural",
    '/home/chuongh/vm2m/data/HIM2K/instmatt_public_r50_fpn_3x/natural',
    '/home/chuongh/vm2m/data/HIM2K/instmatt_r50_fpn_3x/natural',
    '/home/chuongh/vm2m/output/HIM/baselines/mgm_single-mask/mgm_cvpr23_multi-inst_him_bs12_0920/vis_r50_fpn_3x/natural',
    '/home/chuongh/vm2m/output/HIM/baselines/mgm_stacked-mask/mgm_cvpr23_stacked_multi-inst_him_bs12_0920/vis_r50_fpn_3x/natural',
    # '/home/chuongh/vm2m/output/HIM/ours_1106_single-stage_stronger-aug_ft_guidance2/vis_r50_fpn_3x/natural'
    '/home/chuongh/sensei-fs-symlink/users/chuongh/vm2m/output/HIM/ours_1110_stronger-aug_guidance_scratch/vis_24.5k_r50_fpn_3x/natural'
]
pred_dirs = [
    "/home/chuongh/vm2m/data/vis/natural_wo_gt/masks",
    "/home/chuongh/vm2m/data/vis/natural_wo_gt/instmatt_public",
    "/home/chuongh/vm2m/data/vis/natural_wo_gt/instmatt",
    "/home/chuongh/vm2m/data/vis/natural_wo_gt/sparsemat",
    "/home/chuongh/vm2m/data/vis/natural_wo_gt/mgm",
    "/home/chuongh/vm2m/data/vis/natural_wo_gt/mgm_stacked",
    "/home/chuongh/vm2m/data/vis/natural_wo_gt/ours"
]
names = [
    "mask",
    "instmatt-public",
    "instmatt-retrained",
    "sparsemat",
    "mgm-single",
    "mgm-stacked",
    "ours"
]
output_dir = "/home/chuongh/vm2m/output/HIM/qualitative_1124"
os.makedirs(output_dir, exist_ok=True)

# target_files = [
#     "google_easy_ee8f845cad1049d185cadcb0d07c927b",
#     "google_easy_e8c4d9c837774ae1bf05d298d9dd0440",
#     "Pexels_middle_pexels-photo-2131689",
#     "unsplash_middle_sherman-yang-Y8i0X70t4fU-unsplash",
#     "Pexels_middle_pexels-photo-7148409",
#     "google_easy_7a1d95503adf4d868d61c0399dd85bb7",
#     "celebrity_easy_386d46454c8a4519b0dff0775716add0",
#     "google_easy_f4851ee5bcb148e79437653c8d437d49",
#     "Pexels_easy_pexels-photo-6113389",
#     "Pexels_easy_pexels-photo-6383216",
#     "Pexels_middle_pexels-photo-5063068",
#     "unsplash_middle_adam-smith-ebGd-nqQCH8-unsplash",
#     "unsplash_middle_zachary-nelson-98Elr-LIvD8-unsplash"
# ]

target_files = [
    "google_middle_95e6da3a2f164a80b68abdd8058aa696",
    "celebrity_middle_1d3504849baa441fb6367d178c892423",
    "Pexels_easy_pexels-photo-3852148",
    "Pexels_middle_pexels-photo-3228738",
    "Pexels_easy_pexels-photo-1719353",
    "Pexels_easy_pexels-photo-4173142",
    "Pexels_easy_pexels-photo-4820227",
    "Pexels_easy_pexels-photo-5623079",
    "Pexels_easy_pexels-photo-6113389",
    "Pexels_easy_pexels-photo-6519865",
    "Pexels_middle_pexels-photo-2709987",
    "Pexels_middle_pexels-photo-5119588"
]
target_files = [
    "Pexels_hard_pexels-photo-4342428",
    "google_hard_b147dccbce8d44b59593112cf93b6c63",
    "google_hard_b5d28609be75416e9544d2f80a593b5d",
    "unsplash_hard_austin-blanchard-rl4XKyX9UCI-unsplash",
    "google_hard_5e3cbbdbf91e49179a85f2a7d2e1d3d8",
    "Pexels_hard_pexels-photo-4820190",
    "Pexels_hard_pexels-photo-6760896",
    "Pexels_hard_pexels-photo-1815257",
    "Pexels_hard_pexels-photo-3289167"
]
'''
image_name:
|--- image
|--- alpha_0
|--- alpha_1
|---- mask_0
|---- mask_1
|---- instmatt-public_0
....
'''

for image_name in target_files:
    source_image_path = os.path.join(image_root, image_name + '.jpg')
    target_image_path = os.path.join(output_dir, image_name, 'image.jpg')
    # copy image
    os.makedirs(os.path.dirname(target_image_path), exist_ok=True)
    shutil.copy2(source_image_path, target_image_path)

    # Load all alphas
    alpha_names = sorted(os.listdir(os.path.join(pred_dirs[1], image_name)))
    for inst_i, alpha_name in enumerate(alpha_names):
        for outname, pred_dir in zip(names, pred_dirs):
            source_path = os.path.join(pred_dir, image_name, alpha_name)
            target_path = os.path.join(output_dir, image_name, outname + '_' + str(inst_i) + '.png')
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)

    
    
    