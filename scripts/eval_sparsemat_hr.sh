# PORT=1104
# for SUBSET in natural comp
# do
#     for MODEL in r50_c4_3x r50_dc5_3x r50_fpn_3x r50_fpn_400e r101_c4_3x r101_fpn_3x r101_fpn_400e regnetx_400e regnety_400e x101_fpn_3x
#     do
#         python -m tools.main --dist --gpus 4 --dist-url tcp://127.0.0.1:$PORT --config configs/HIM/sparsemat_hr_test.yaml --eval-only \
#                                                         name sparsemat_hr_1102 \
#                                                         model.weights output/HIM/sparsemat_cvpr23_him_1152_bs16_1030/best_model.pth \
#                                                         dataset.test.split $SUBSET \
#                                                         dataset.test.downscale_mask False \
#                                                         dataset.test.mask_dir_name masks_matched_$MODEL \
#                                                         test.save_results False \
#                                                         test.postprocessing False \
#                                                         test.use_trimap True \
#                                                         test.temp_aggre False \
#                                                         test.log_iter 10 \
#                                                         dataset.test.root_dir /mnt/localssd/HIM2K \
#                                                         test.save_dir output/HIM/sparsemat_cvpr23_him_bs64_0920/vis
#     done
# done

for NO_INST in 1 2 3 4 5 6 7 8 9 10
do
    export NO_INST=$NO_INST
    python -m tools.main --config configs/HIM/sparsemat_hr_test.yaml --eval-only \
                                                    name sparsemat_hr_1102 \
                                                    model.weights output/HIM/baselines/sparsemat/sparsemat_cvpr23_him_1152_bs16_1030/best_model.pth \
                                                    name benchmark_sparsemat \
                                                    output_dir output/HIM \
                                                    dataset.test.split benchmark \
                                                    dataset.test.downscale_mask False \
                                                    dataset.test.mask_dir_name masks_matched_r50_fpn_3x \
                                                    test.save_results False \
                                                    test.postprocessing False \
                                                    test.use_trimap True \
                                                    test.temp_aggre False \
                                                    test.log_iter 10 \
                                                    dataset.test.root_dir /mnt/localssd/HIM2K
done     

python -m tools.main --config configs/HIM/sparsemat_hr_test.yaml --eval-only \
                                                    name sparsemat_hr_1102 \
                                                    model.weights output/HIM/baselines/sparsemat/sparsemat_cvpr23_him_1152_bs16_1030/best_model.pth \
                                                    name benchmark_sparsemat \
                                                    output_dir output/HIM \
                                                    dataset.test.split natural_wo_gt \
                                                    dataset.test.downscale_mask False \
                                                    dataset.test.mask_dir_name masks_r101_fpn_3x \
                                                    dataset.test.alpha_dir_name masks_r101_fpn_3x \
                                                    test.save_results True \
                                                    test.save_dir /home/chuongh/vm2m/data/vis/natural_wo_gt/sparsemat \
                                                    test.postprocessing False \
                                                    test.use_trimap True \
                                                    test.temp_aggre False \
                                                    dataset.test.root_dir /mnt/localssd/HIM2K \
                                                    test.log_iter 10