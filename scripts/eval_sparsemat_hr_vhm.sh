export CUDA_VISIBLE_DEVICES=1
for SUBSET in comp_easy comp_medium comp_hard real
do
    python -m tools.main --eval-only --config output/VHM/sparsemat_hr_1101/config.yaml \
        dataset.test.mask_dir_name xmem \
        dataset.test.split $SUBSET \
        model.weights output/VHM/sparsemat_hr_1101/best_model.pth \
        test.save_results True \
        test.save_dir output/VHM/sparsemat_hr_1101/vis/$SUBSET
done