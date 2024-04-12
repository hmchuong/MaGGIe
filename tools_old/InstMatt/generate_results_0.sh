export CUDA_VISIBLE_DEVICES=0
for SUBSET in natural comp
do
    for MODEL in r50_c4_3x r50_dc5_3x
    do
        python infer_0814.py --config config/InstMatt-stage2.toml --checkpoint checkpoints/best_model.pth \
            --image-pattern '/mnt/localssd/HIM2K/images/'${SUBSET}'/*.jpg' \
            --mask-dir /mnt/localssd/HIM2K/masks_matched_${MODEL}/${SUBSET} \
            --output /mnt/localssd/HIM2K/instmatt_public_${MODEL}/${SUBSET}
    done
done