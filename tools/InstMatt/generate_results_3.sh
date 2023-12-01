export CUDA_VISIBLE_DEVICES=3
for SUBSET in natural comp
do
    for MODEL in regnety_400e x101_fpn_3x
    do
        python infer_0814.py --config config/InstMatt-stage2.toml --checkpoint checkpoints/best_model.pth \
            --image-pattern '/mnt/localssd/HIM2K/images/'${SUBSET}'/*.jpg' \
            --mask-dir /mnt/localssd/HIM2K/masks_matched_${MODEL}/${SUBSET} \
            --output /mnt/localssd/HIM2K/instmatt_public_${MODEL}/${SUBSET}
    done
done