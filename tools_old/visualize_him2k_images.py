import os
import numpy as np
import cv2
import tqdm

# For masks = 0.5
# image_root = "/home/chuongh/vm2m/data/vis/natural_wo_gt/images"
image_root = "/home/chuongh/vm2m/data/vis/generalization/images"
# pred_dirs = [
#     "/mnt/localssd/HIM2K/alphas/natural",
#     "/home/chuongh/vm2m/output/HIM/ours_1102_single-stage_strong-aug_ft/debug/natural",
#     "/home/chuongh/sensei-fs-symlink/users/chuongh/vm2m/output/HIM/mgm_cvpr23_stacked_multi-inst_him_bs12_0920/vis/alpha_pred/natural"
# ]

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
    "/home/chuongh/vm2m/data/vis/natural_wo_gt/instmatt",
    "/home/chuongh/vm2m/data/vis/natural_wo_gt/sparsemat",
    "/home/chuongh/vm2m/data/vis/natural_wo_gt/mgm",
    "/home/chuongh/vm2m/data/vis/natural_wo_gt/mgm_stacked",
    "/home/chuongh/vm2m/data/vis/natural_wo_gt/ours"
]

pred_dirs = [
    "/home/chuongh/vm2m/data/vis/generalization/masks",
    "/home/chuongh/vm2m/data/vis/generalization/ours"
]
output_dir = "/home/chuongh/vm2m/output/HIM/combine_him2k_generalization_1122_all_green"
os.makedirs(output_dir, exist_ok=True)

# target_files = [
#     "unsplash_middle_saeed-karimi-JrrWC7Qcmhs-unsplash",
#     "google_easy_c3668dda4d46436097b6c5d58153e7de",
#     "google_middle_95e6da3a2f164a80b68abdd8058aa696",
#     "google_middle_62604586fc08499ca4d346900c338bc8",
#     "google_middle_44440b07b2d5459ea2053cd3ee9f9406",
#     "google_middle_2db0e2572a654f55947d88d411c3cb61",
#     "celebrity_middle_d3874137945e41319d93448910b8bcdd",
#     "celebrity_easy_e129df9159a64841b62c588ddc5730c9",
#     "celebrity_middle_1650c9fd73ae4313bae6a598a7d7072e",
#     "celebrity_middle_8f47dc49975f4fc484224fdec092253f",
#     "celebrity_easy_c0c60daf04cf41e3bb5752d932944dd9",
#     "celebrity_middle_2b9db99c95e54d918f2a0795515ef271",
#     "celebrity_middle_1d3504849baa441fb6367d178c892423",
#     "celebrity_easy_b71d8703a11240a2aba6babc1193c2e4",
#     "Pexels_middle_pexels-photo-5896435",
#     "Pexels_easy_pexels-photo-5618157"
# ]

# target_files = [
#     "google_middle_1531bdee643d477984262adf9cbde17c",
#     "google_middle_8418b4434e9a4490b573be160f0d3c03",
#     "google_middle_44440b07b2d5459ea2053cd3ee9f9406",
#     "google_middle_07d854895200416387ff3702c453f12c",
#     "google_easy_d286b3dd0f124236a649dad67a3c8095",
#     "google_easy_4733e069037b45c89012d38e02b0a676",
#     "google_easy_7a1d95503adf4d868d61c0399dd85bb7",
#     "celebrity_middle_d3874137945e41319d93448910b8bcdd",
#     "celebrity_middle_654c266aa15749e1893a9c40c4f79ad5",
#     "celebrity_middle_1d3504849baa441fb6367d178c892423",
#     "celebrity_middle_62fd434bc6ca4d359e6260763d7c6716",
#     "celebrity_middle_00ae4a6bd5a94cbab28d9f7abed7e6bd",
#     "celebrity_easy_e129df9159a64841b62c588ddc5730c9",
#     "celebrity_easy_9768f2fb1df94cf7ad5bed17373c3108",
#     "celebrity_easy_864a5a3ddb5445bcb2a9b2a929852e73",
#     "celebrity_easy_c0c60daf04cf41e3bb5752d932944dd9",
#     "Pexels_middle_pexels-photo-1157536",
#     "Pexels_middle_pexels-photo-1393509",
#     "Pexels_middle_pexels-photo-1502874",
#     "Pexels_middle_pexels-photo-8223988"
# ]

# target_files = [
#     "celebrity_easy_28545f023c9c4f22ad2f16228baaad3d",
#     "celebrity_easy_7ce70553b6ca4dc58d53cc0ca4edad11",
#     "Pexels_middle_pexels-photo-8224679",
#     "unsplash_easy_clarisse-meyer-UISgcA0yLrA-unsplash",
#     "google_easy_5a59db72c81b4370be71c9030c3bf69f",
#     "google_middle_2598f2022a5243b5ae172f346266befa",
#     "Pexels_middle_pexels-photo-6140614",
#     "celebrity_easy_f42a3d17cea3454199551718df17ef09",
#     "celebrity_easy_ab967ceddc2d452eb3f0ba6319729f6b",
#     "google_easy_11771c8bd83944f8a5e4bd3dce899fa1",
#     "Pexels_easy_pexels-photo-5256609",
#     "unsplash_easy_evelyn-cespedes-QLSov-QKfWM-unsplash",
#     "celebrity_easy_adca06887f6f42219846edf97f0d8f89",
#     "Pexels_easy_pexels-photo-3259971",
#     "Pexels_easy_pexels-photo-4746207",
#     "Pexels_middle_pexels-photo-1212805",
#     "Pexels_middle_pexels-photo-5119526",
#     "google_easy_11733cb3b561440da199c4b753122f52",
#     "Pexels_middle_pexels-photo-6113385",
#     "Pexels_middle_pexels-photo-8217914",
#     "google_middle_0debd6d76d2840ccbc2f30fa42218664",
#     "Pexels_middle_pexels-photo-1739842",
#     "unsplash_middle_sherman-yang-Y8i0X70t4fU-unsplash",
#     "google_easy_0ba90044a0bc413991220e6cfa5a4a1b",
#     "google_easy_3dcb90547b1847e2ad040b06b4bfb78e",
#     "Pexels_middle_pexels-photo-3791666",
#     "Pexels_easy_pexels-photo-6774874",
#     "unsplash_easy_the-creative-exchange-kpUWtrP3_Q8-unsplash",
#     "Pexels_easy_pexels-photo-5317718",
#     "Pexels_easy_pexels-photo-5553031",
#     "google_easy_6488fdb104ea41bb8c7030fd4c0ef0e6",
#     "Pexels_middle_pexels-photo-6140970",
#     "celebrity_easy_9768f2fb1df94cf7ad5bed17373c3108",
#     "celebrity_middle_2b9db99c95e54d918f2a0795515ef271",
#     "celebrity_easy_c82c8eb3e49e421bb6635f1afe2aa847",
#     "Pexels_middle_pexels-photo-4018835",
#     "celebrity_easy_1118950f083e4f8fb84c35c834d304ff",
#     "celebrity_easy_e381dc135e1c48869c27860faee1bc2a",
#     "Pexels_easy_pexels-photo-8364643",
#     "celebrity_middle_654c266aa15749e1893a9c40c4f79ad5",
#     "Pexels_middle_pexels-photo-4473806",
#     "Pexels_middle_pexels-photo-2062813",
#     "google_easy_f4851ee5bcb148e79437653c8d437d49",
#     "celebrity_easy_e070e9ca489c40daac0babfef80583b2",
#     "Pexels_middle_pexels-photo-5623730",
#     "google_middle_86d3edc1c16f48d5880c9448ae2f4606",
#     "Pexels_middle_pexels-photo-3611850",
#     "celebrity_easy_15f66d3f34ef4cc0b53809222fe1eb68",
#     "google_middle_7c5ff238fe104c279e9b069c20888f0d",
#     "Pexels_easy_pexels-photo-4918465",
#     "Pexels_middle_pexels-photo-1094085",
#     "Pexels_middle_pexels-photo-301977",
#     "google_easy_4733e069037b45c89012d38e02b0a676",
#     "Pexels_easy_pexels-photo-5628760",
#     "Pexels_middle_pexels-photo-936058",
#     "Pexels_middle_pexels-photo-4787527",
#     "Pexels_middle_pexels-photo-5212659",
#     "google_middle_8418b4434e9a4490b573be160f0d3c03",
#     "unsplash_middle_saeed-karimi-JrrWC7Qcmhs-unsplash"
# ]
# target_files = [
#     "Pexels_middle_pexels-photo-1393509",
# "google_middle_2e2db6b037aa4f61a70f32904e52e7d7",
# "Pexels_middle_pexels-photo-1502874",
# "celebrity_easy_cef5415a401c4c20b9dff7620fa2dd99",
# "Pexels_middle_pexels-photo-1325047",
# "google_middle_2db0e2572a654f55947d88d411c3cb61",
# "google_easy_22b3ffecdf594a75a765e25b3e4ccda8",
# "google_easy_b36f65b827254a129da6347a7a0faf28",
# "celebrity_middle_8f47dc49975f4fc484224fdec092253f",
# "google_easy_c3668dda4d46436097b6c5d58153e7de",
# "google_middle_44440b07b2d5459ea2053cd3ee9f9406",
# "google_easy_7cb3473a7b20434781a344a10f0b7408",
# "Pexels_middle_pexels-photo-2062813",
# "google_middle_5e4e4a7008fe4f948cf370b94f1dc0b1",
# "Pexels_middle_pexels-photo-1586039",
# "celebrity_middle_1d3504849baa441fb6367d178c892423",
# "Pexels_middle_pexels-photo-2647097",
# "Pexels_middle_pexels-photo-2761616",
# "Pexels_easy_pexels-photo-4308050",
# "Pexels_middle_pexels-photo-3775551",
# "Pexels_middle_pexels-photo-1475012",
# "google_middle_63165b6e4e354f418983895aea8ea467",
# "Pexels_easy_pexels-photo-4557514",
# "google_middle_0138f6dd36e74a49bc89ab035b1f8c7c",
# "celebrity_easy_e129df9159a64841b62c588ddc5730c9",
# "Pexels_easy_pexels-photo-5618157",
# "google_middle_62604586fc08499ca4d346900c338bc8",
# "google_middle_95e6da3a2f164a80b68abdd8058aa696",
# "Pexels_middle_pexels-photo-5119588",
# "Pexels_middle_pexels-photo-5896435",
# "Pexels_middle_pexels-photo-1421769",
# "Pexels_middle_pexels-photo-1619706",
# "Pexels_middle_pexels-photo-708392",
# "Pexels_middle_pexels-photo-6140614",
# "Pexels_easy_pexels-photo-6774281",
# "unsplash_middle_tony-mucci-3i_88K3N0Mc-unsplash",
# "Pexels_middle_pexels-photo-6140970",
# "celebrity_middle_24af2a17156c48c0a7bb067f9c414b63",
# "Pexels_middle_pexels-photo-2385577",
# "Pexels_middle_pexels-photo-1212805",
# "celebrity_easy_c0c60daf04cf41e3bb5752d932944dd9",
# "Pexels_middle_pexels-photo-1326065",
# "Pexels_middle_pexels-photo-2161298",
# "google_easy_36c9106c9fc748748eabc80e4730184b",
# "celebrity_middle_60e0a86a206d424e81af662be7c865a3",
# "google_easy_4e5457e1bbf5428d97e12702c6d9c7b4",
# "google_easy_4ac7d734f5b64db49072b7b186318b64",
# "celebrity_middle_654c266aa15749e1893a9c40c4f79ad5",
# "celebrity_easy_136486f535f04470a2e5fd12a1a9795e",
# "celebrity_middle_62fd434bc6ca4d359e6260763d7c6716"
# ]

target_files = [
    "google_middle_07d854895200416387ff3702c453f12c",
"celebrity_middle_d3874137945e41319d93448910b8bcdd",
"celebrity_middle_00ae4a6bd5a94cbab28d9f7abed7e6bd",
"celebrity_middle_0e215bcb71b54976ab4fca0cf64f8e0e",
"Pexels_middle_pexels-photo-6140614",
"google_middle_2e2db6b037aa4f61a70f32904e52e7d7",
"Pexels_middle_pexels-photo-6140970",
"unsplash_middle_saeed-karimi-JrrWC7Qcmhs-unsplash",
"unsplash_middle_sherman-yang-Y8i0X70t4fU-unsplash",
"Pexels_middle_pexels-photo-1586039",
"Pexels_middle_pexels-photo-3890209",
"celebrity_middle_1d3504849baa441fb6367d178c892423",
"celebrity_easy_ab967ceddc2d452eb3f0ba6319729f6b",
"Pexels_easy_pexels-photo-4557514",
"celebrity_easy_1dfc7b119b4e4fa0b5aea369b52883bc",
"google_easy_e8c4d9c837774ae1bf05d298d9dd0440",
"google_middle_0138f6dd36e74a49bc89ab035b1f8c7c",
"Pexels_middle_pexels-photo-4018835",
"Pexels_middle_pexels-photo-8217914",
"google_easy_352d4181e4314a9796b5ac63c5233f38",
"Pexels_middle_pexels-photo-3228712",
"Pexels_easy_pexels-photo-3958844",
"Pexels_middle_pexels-photo-2131689",
"Pexels_middle_pexels-photo-3051569",
"google_middle_86d3edc1c16f48d5880c9448ae2f4606",
"Pexels_easy_pexels-photo-2624875",
"Pexels_easy_pexels-photo-8364643",
"google_easy_3dcb90547b1847e2ad040b06b4bfb78e",
"Pexels_middle_pexels-photo-1739842",
"Pexels_middle_pexels-photo-6246574",
"Pexels_middle_pexels-photo-1475012",
"google_easy_d286b3dd0f124236a649dad67a3c8095",
"Pexels_easy_pexels-photo-1166990",
"Pexels_middle_pexels-photo-4880395",
"google_middle_56740c7706a142e7822d04907240c44f",
"Pexels_easy_pexels-photo-5256609",
"unsplash_middle_amy-kate-Xsppd5V1yKE-unsplash",
"google_easy_5a59db72c81b4370be71c9030c3bf69f",
"celebrity_easy_2238c65c4a2c40e58ba90a0357f98a9c",
"unsplash_easy_vince-fleming-9PIfZHcnrjQ-unsplash",
"Pexels_middle_pexels-photo-1140916",
"celebrity_easy_f7d308ee9cbc42d3973657c647c32f2e",
"Pexels_middle_pexels-photo-3611850",
"Pexels_middle_pexels-photo-52578",
"Pexels_easy_pexels-photo-7148445",
"Pexels_easy_pexels-photo-5553031",
"google_easy_42fbb3c0abaf4fe2807db58090a39f45",
"google_easy_14f7900ee1d141d28dcf935010875cd3",
"Pexels_easy_pexels-photo-5834140",
"Pexels_easy_pexels-photo-1197373",
"unsplash_easy_fabien-bazanegue-FVv0P5O6PC0-unsplash",
"Pexels_middle_pexels-photo-7148409",
"Pexels_middle_pexels-photo-5384613",
"Pexels_middle_pexels-photo-2332083",
"Pexels_middle_pexels-photo-5119526",
"unsplash_easy_clarisse-meyer-UISgcA0yLrA-unsplash",
"Pexels_middle_pexels-photo-5124464",
"google_easy_ebbc4ce2f2c44b0392b27b9e9769c7a4",
"google_easy_360567395a414461bea878df4f587299",
"Pexels_middle_pexels-photo-4473806",
"Pexels_easy_pexels-photo-3585812",
"google_easy_4ac7d734f5b64db49072b7b186318b64",
"Pexels_easy_pexels-photo-6787440",
"Pexels_easy_pexels-photo-5623684",
"Pexels_middle_pexels-photo-7915402",
"Pexels_easy_pexels-photo-4880350",
"Pexels_easy_pexels-photo-5628760",
"Pexels_easy_pexels-photo-4918791",
"Pexels_easy_pexels-photo-1361670",
"google_easy_ee0b5bf0b0f7490f9e43e80a4b2fc371"
]

for image_name in tqdm.tqdm(os.listdir(image_root)):
# for image_name in tqdm.tqdm(target_files):
    # image_name = image_name + ".jpg"
    image = cv2.imread(os.path.join(image_root, image_name))
    # print(image_root, image_name)
    h, w, _ = image.shape
    w = int(2000 / h * w)
    image = cv2.resize(image, (w, 2000))

    # Load all alphas
    alpha_names = sorted(os.listdir(os.path.join(pred_dirs[1], image_name.replace(".jpg", ""))))
    instance_h = 2000 // len(alpha_names)
    instance_w = int(w * instance_h / 2000)

    new_image = np.zeros((2000, w + instance_w * len(pred_dirs), 3), dtype=np.uint8)
    new_image[:, :w] = image
    x = w
    y = 0
    
    for inst_i, alpha_name in enumerate(alpha_names):
        alpha_gt = None
        for i, pred_dir in enumerate(pred_dirs):
            mask = None
            if os.path.exists(os.path.join(pred_dir, image_name.replace(".jpg", ""), alpha_name)):
                mask = cv2.imread(os.path.join(pred_dir, image_name.replace(".jpg", ""), alpha_name))
            else:
                mask = cv2.imread(os.path.join(pred_dir, image_name.replace(".jpg", f"_inst{inst_i}.jpg")))
            mask = cv2.resize(mask, (instance_w, instance_h))
            frame = cv2.resize(image, (instance_w, instance_h))
            green_bg = np.zeros_like(frame)
            green_bg[:, :, 1] = 255
            mask = mask.astype(np.float32) / 255
            mask = mask * frame + (1 - mask) * green_bg
            if len(mask.shape) == 1:
                mask = np.expand_dims(mask, axis=-1)
                mask = np.concatenate([mask, mask, mask], axis=-1)
            mask[:2, :, 2] = 255
            mask[-2:, :, 2] = 255
            mask[:, :2, 2] = 255
            mask[:, -2:, 2] = 255
            new_image[y: y + instance_h, x + (instance_w * i):x + (instance_w * (i + 1))] = mask
        y += instance_h
    cv2.imwrite(os.path.join(output_dir, image_name), new_image)