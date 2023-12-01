
for SUBSET in natural comp
do 
    python image_demo_newbaselines.py --config-file $1 --input /home/chuongh/vm2m/data/HIM2K/images/$SUBSET/*.jpg --output /home/chuongh/vm2m/data/HIM2K/masks_$2/$SUBSET --opts MODEL.WEIGHTS ../pretrained/model_$2.pkl
done

python image_demo_newbaselines.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml --input /home/chuongh/vm2m/data/HIM2K/images/natural_wo_gt/*.jpg --output /home/chuongh/vm2m/data/HIM2K/masks_r101_fpn_3x/natural_wo_gt --opts MODEL.WEIGHTS ../pretrained/model_r101_fpn_3x.pkl