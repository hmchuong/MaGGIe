for SUBSET in natural comp
do
    for MODEL in r50_c4_3x r50_dc5_3x r50_fpn_3x r50_fpn_400e r101_c4_3x r101_fpn_3x r101_fpn_400e regnetx_400e regnety_400e x101_fpn_3x
    do
    python -m tools.main --dist --gpus 4 --dist-url tcp://127.0.0.1:1234 --config $1 --eval-only \
                                                    name m-him2k \
                                                    model.weights $2 \
                                                    dataset.test.split $SUBSET \
                                                    dataset.test.downscale_mask False \
                                                    dataset.test.mask_dir_name masks_matched_${MODEL} \
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
#                                                     model.weights output/HIM/$1/best_model.pth \
#                                                     dataset.test.split combine \
#                                                     dataset.test.downscale_mask False \
#                                                     dataset.test.mask_dir_name masks \
#                                                     test.save_results False \
#                                                     test.postprocessing False \
#                                                     test.use_trimap True \
#                                                     test.temp_aggre False \
#                                                     test.log_iter 10

# python -m tools.main --dist --gpus 4 --dist-url tcp://127.0.0.1:1100 --config output/HIM/ours_1110_stronger-aug_guidance_scratch/config.yaml --eval-only \
#                                                     model.weights output/HIM/ours_1110_stronger-aug_guidance_scratch/last_model_24k.pth \
#                                                     dataset.test.split test \
#                                                     dataset.test.downscale_mask False \
#                                                     dataset.test.mask_dir_name masks \
#                                                     dataset.test.short_size 512 \
#                                                     test.save_results True \
#                                                     test.save_dir output/HIM/ours_1110_stronger-aug_guidance_scratch/test \
#                                                     test.postprocessing False \
#                                                     test.use_trimap True \
#                                                     test.temp_aggre False \
#                                                     test.log_iter 10

# for NO_INST in 1 2 3 4 5 6 7 8 9 10
# do 
#     export NO_INST=$NO_INST
#     python -m tools.main --config output/HIM/ours_1110_stronger-aug_guidance_scratch/config.yaml --eval-only \
#                                                         model.weights output/HIM/ours_1110_stronger-aug_guidance_scratch/last_model_24k.pth \
#                                                         name benchmark_ours \
#                                                         output_dir output/HIM \
#                                                         dataset.test.split benchmark \
#                                                         dataset.test.downscale_mask False \
#                                                         dataset.test.mask_dir_name masks_matched_r50_fpn_3x \
#                                                         test.save_results False \
#                                                         test.postprocessing False \
#                                                         test.use_trimap True \
#                                                         test.temp_aggre False \
#                                                         test.log_iter 10
# done         

# python -m tools.main --config output/HIM/ours_1110_stronger-aug_guidance_scratch/config.yaml --eval-only \
#                                                         model.weights output/HIM/ours_1110_stronger-aug_guidance_scratch/last_model_24k.pth \
#                                                         name qualitative_ours \
#                                                         output_dir output/HIM \
#                                                         dataset.test.split natural_wo_gt \
#                                                         dataset.test.downscale_mask False \
#                                                         dataset.test.mask_dir_name masks_r101_fpn_3x \
#                                                         dataset.test.alpha_dir_name masks_r101_fpn_3x \
#                                                         test.save_results True \
#                                                         test.save_dir /home/chuongh/vm2m/data/vis/natural_wo_gt/ours \
#                                                         test.postprocessing False \
#                                                         test.use_trimap True \
#                                                         test.temp_aggre False \
#                                                         test.log_iter 10

# python -m tools.main --config output/HIM/ours_1110_stronger-aug_guidance_scratch/config.yaml --eval-only \
#                                                         model.weights output/HIM/ours_1110_stronger-aug_guidance_scratch/last_model_24k.pth \
#                                                         name qualitative_ours \
#                                                         output_dir output/HIM \
#                                                         dataset.test.split generalization \
#                                                         dataset.test.downscale_mask False \
#                                                         dataset.test.mask_dir_name masks \
#                                                         dataset.test.alpha_dir_name masks \
#                                                         test.save_results True \
#                                                         test.save_dir /home/chuongh/vm2m/data/vis/generalization \
#                                                         test.postprocessing False \
#                                                         test.use_trimap True \
#                                                         test.temp_aggre False \
#                                                         test.log_iter 10                                                        