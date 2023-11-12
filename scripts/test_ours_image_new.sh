for SUBSET in natural comp #comp
do
    for MODEL in r50_c4_3x r50_dc5_3x r50_fpn_3x r50_fpn_400e r101_c4_3x r101_fpn_3x r101_fpn_400e regnetx_400e regnety_400e x101_fpn_3x
    do
    python -m tools.main --dist --gpus 4 --dist-url tcp://127.0.0.1:$2 --config output/HIM/$1/config.yaml --eval-only \
                                                    name ${1}_new_mask \
                                                    model.weights output/HIM/$1/last_model_24.5k.pth \
                                                    dataset.test.split $SUBSET \
                                                    dataset.test.downscale_mask False \
                                                    dataset.test.mask_dir_name masks_matched_${MODEL}_new \
                                                    test.save_results False \
                                                    test.postprocessing False \
                                                    test.use_trimap True \
                                                    test.temp_aggre False \
                                                    test.log_iter 10
    done
done
# python -m tools.main --dist --gpus 4 --dist-url tcp://127.0.0.1:1100 --config output/HIM/ours_1103_single-stage_strong-aug/config.yaml --eval-only \
#                                                     model.weights output/HIM/ours_1103_single-stage_strong-aug/best_model.pth \
#                                                     dataset.test.split natural \
#                                                     dataset.test.downscale_mask False \
#                                                     dataset.test.mask_dir_name masks_matched_r50_fpn_3x \
#                                                     test.save_dir output/HIM/ours_1103_single-stage_strong-aug/vis \
#                                                     test.save_results True \
#                                                     test.postprocessing False \
#                                                     test.use_trimap True \
#                                                     test.temp_aggre False \
#                                                     test.log_iter 10


# python -m tools.main --dist --gpus 4 --dist-url tcp://127.0.0.1:$2 --config output/HIM/$1/config.yaml --eval-only \
#                                                     name $1 \
#                                                     model.weights output/HIM1/$1/best_model.pth \
#                                                     dataset.test.split combine \
#                                                     dataset.test.downscale_mask False \
#                                                     dataset.test.mask_dir_name masks \
#                                                     test.save_results False \
#                                                     test.postprocessing False \
#                                                     test.use_trimap True \
#                                                     test.temp_aggre False \
#                                                     test.log_iter 10