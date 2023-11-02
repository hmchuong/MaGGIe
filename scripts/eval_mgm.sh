PORT=1101
for SUBSET in natural comp
do
    for MODEL in r50_c4_3x r50_dc5_3x r50_fpn_3x r50_fpn_400e r101_c4_3x r101_fpn_3x r101_fpn_400e regnetx_400e regnety_400e x101_fpn_3x
    do
        python -m tools.main --dist --dist-url tcp://127.0.0.1:$PORT --gpus 4 --config output/HIM/ours_short-576-512x512_bs12_nnodes-2_gpus-4_52k_adamw_1.5e-4_0903/config.yaml --eval-only \
                                                        name mgm_retrain_1102 \
                                                        model.arch MGM_SingInst \
                                                        model.backbone res_shortcut_encoder_29 \
                                                        model.decoder res_shortcut_decoder_22 \
                                                        model.backbone_args.num_mask 1 \
                                                        model.decoder_args.max_inst 1 \
                                                        model.weights output/HIM/mgm_cvpr23_him_bs64_0919/best_model.pth \
                                                        dataset.test.split $SUBSET \
                                                        dataset.test.downscale_mask False \
                                                        dataset.test.mask_dir_name masks_matched_$MODEL \
                                                        test.save_results False \
                                                        test.postprocessing False \
                                                        test.use_trimap True \
                                                        test.temp_aggre False \
                                                        dataset.test.root_dir /mnt/localssd/HIM2K \
                                                        test.log_iter 10
    done
done