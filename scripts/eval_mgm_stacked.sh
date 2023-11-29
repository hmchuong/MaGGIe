# PORT=1102
# for SUBSET in natural comp
# do
#     for MODEL in r50_c4_3x r50_dc5_3x r50_fpn_3x r50_fpn_400e r101_c4_3x r101_fpn_3x r101_fpn_400e regnetx_400e regnety_400e x101_fpn_3x
#     do
#         python -m tools.main --dist --dist-url tcp://127.0.0.1:$PORT --gpus 4 --config output/HIM/ours_short-576-512x512_bs12_nnodes-2_gpus-4_52k_adamw_1.5e-4_0903/config.yaml --eval-only \
#                                                     name mgm_stacked_1020 \
#                                                     model.arch MGM \
#                                                     model.backbone res_shortcut_encoder_29 \
#                                                     model.decoder res_shortcut_decoder_22 \
#                                                     model.backbone_args.num_mask 10 \
#                                                     model.decoder_args.max_inst 10 \
#                                                     model.weights output/HIM/mgm_cvpr23_stacked_multi-inst_him_bs12_0920/best_model.pth \
#                                                     dataset.test.split $SUBSET \
#                                                     dataset.test.downscale_mask False \
#                                                     dataset.test.mask_dir_name masks_matched_$MODEL \
#                                                     test.save_results False \
#                                                     test.postprocessing False \
#                                                     test.use_trimap True \
#                                                     test.temp_aggre False \
#                                                     dataset.test.root_dir /mnt/localssd/HIM2K/benchmark \
#                                                     test.log_iter 10
#     done
# done

for NO_INST in 1 2 3 4 5 6 7 8 9 10
do 
    export NO_INST=$NO_INST
    python -m tools.main --config output/HIM/baselines/mgm_stacked-mask/mgm_cvpr23_stacked_multi-inst_him_bs12_0920/config.yaml --eval-only \
                                                        name benchmark_mgm_stacked_1020 \
                                                        output_dir output/HIM \
                                                        model.arch MGM \
                                                        model.backbone res_shortcut_encoder_29 \
                                                        model.decoder res_shortcut_decoder_22 \
                                                        model.backbone_args.num_mask 10 \
                                                        model.decoder_args.max_inst 10 \
                                                        model.weights output/HIM/baselines/mgm_stacked-mask/mgm_cvpr23_stacked_multi-inst_him_bs12_0920/best_model.pth \
                                                        dataset.test.split benchmark \
                                                        dataset.test.downscale_mask False \
                                                        dataset.test.mask_dir_name masks_matched_r50_fpn_3x \
                                                        test.save_results False \
                                                        test.postprocessing False \
                                                        test.use_trimap True \
                                                        test.temp_aggre False \
                                                        dataset.test.root_dir /mnt/localssd/HIM2K \
                                                        test.log_iter 10
done

python -m tools.main --config output/HIM/baselines/mgm_stacked-mask/mgm_cvpr23_stacked_multi-inst_him_bs12_0920/config.yaml --eval-only \
                                                    name benchmark_mgm_stacked_1020 \
                                                    output_dir output/HIM \
                                                    model.arch MGM \
                                                    model.backbone res_shortcut_encoder_29 \
                                                    model.decoder res_shortcut_decoder_22 \
                                                    model.backbone_args.num_mask 10 \
                                                    model.decoder_args.max_inst 10 \
                                                    model.weights output/HIM/baselines/mgm_stacked-mask/mgm_cvpr23_stacked_multi-inst_him_bs12_0920/best_model.pth \
                                                    dataset.test.split natural_wo_gt \
                                                    dataset.test.downscale_mask False \
                                                    dataset.test.mask_dir_name masks_r101_fpn_3x \
                                                    dataset.test.alpha_dir_name masks_r101_fpn_3x \
                                                    test.save_results True \
                                                    test.save_dir /home/chuongh/vm2m/data/vis/natural_wo_gt/mgm_stacked \
                                                    test.postprocessing False \
                                                    test.use_trimap True \
                                                    test.temp_aggre False \
                                                    dataset.test.root_dir /mnt/localssd/HIM2K \
                                                    test.log_iter 10