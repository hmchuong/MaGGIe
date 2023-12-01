import os
root_dir = "/mnt/localssd/matting"
out_f = open("count_people.sh", 'w')
for image_dir in ["pexels", "unsplash"]:
    for subdir in os.listdir(os.path.join(root_dir, image_dir)):
        out_f.write(f'python count_people_newbaselines.py --config-file ../configs/new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py --input "/mnt/localssd/matting/{image_dir}/{subdir}/*" --output {image_dir}_{subdir}.txt --opts MODEL.WEIGHTS ../pretrained/model_r101_fpn_400e.pkl\n')
out_f.close()