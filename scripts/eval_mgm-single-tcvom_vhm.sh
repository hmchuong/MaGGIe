# export CUDA_VISIBLE_DEVICES=0
# for SUBSET in comp_easy comp_medium comp_hard real
# do
#     python -m tools.main --eval-only --config output/VHM/mgm-single-tcvom_1102/config.yaml \
#         dataset.test.mask_dir_name xmem \
#         dataset.test.split $SUBSET \
#         model.weights output/VHM/mgm-single-tcvom_1102/best_model.pth \
#         test.save_results True \
#         test.save_dir output/VHM/mgm-single-tcvom_1102/vis/$SUBSET
# done

python -m tools.main --eval-only --config output/VHM/mgm-single-tcvom_1102/config.yaml \
        dataset.test.mask_dir_name xmem \
        dataset.test.split real_mosaic \
        model.weights output/VHM/mgm-single-tcvom_1102/best_model.pth \
        test.save_results True \
        test.save_dir output/VHM/mgm-single-tcvom_1102/vis/real_mosaic