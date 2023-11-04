for SUBSET in natural comp
do
    for MODEL in r50_c4_3x r50_dc5_3x r50_fpn_3x r50_fpn_400e r101_c4_3x r101_fpn_3x r101_fpn_400e regnetx_400e regnety_400e x101_fpn_3x
    do
    python -m tools.main --dist --gpus 4 --dist-url tcp://127.0.0.1:$2 --config output/HIM/$1/config.yaml --eval-only \
                                                    name $1 \
                                                    model.weights output/HIM/$1/best_model.pth \
                                                    dataset.test.split $SUBSET \
                                                    dataset.test.downscale_mask False \
                                                    dataset.test.mask_dir_name masks_matched_$MODEL \
                                                    test.save_results False \
                                                    test.postprocessing False \
                                                    test.use_trimap True \
                                                    test.temp_aggre False \
                                                    test.log_iter 10
    done
done