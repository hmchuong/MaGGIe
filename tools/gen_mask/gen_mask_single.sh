
for SUBSET in natural comp
do 
    python image_demo_newbaselines.py --config-file $1 --input data/HIM2K/images/$SUBSET/*.jpg --output data/HIM2K/masks_$2/$SUBSET --opts MODEL.WEIGHTS ../pretrained/model_$2.pkl
done