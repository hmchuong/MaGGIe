OUTPUT_DIR=output/image
NAME=eval_full_$3
for SUBSET in natural comp
do
    for MODEL in r50_c4_3x r50_dc5_3x r50_fpn_3x r50_fpn_400e r101_c4_3x r101_fpn_3x r101_fpn_400e regnetx_400e regnety_400e x101_fpn_3x
    do
    torchrun --standalone --nproc_per_node=$2 tools/main.py --config $1 --eval-only \
                                                    name $NAME \
                                                    output_dir $OUTPUT_DIR \
                                                    dataset.test.split $SUBSET \
                                                    dataset.test.downscale_mask False \
                                                    dataset.test.mask_dir_name masks_matched_${MODEL} \
                                                    test.save_results False \
                                                    test.postprocessing False \
                                                    test.log_iter 10
    done
done

# Write all results to a single csv file
python tools/extract_results.py $OUTPUT_DIR/${NAME}_$3/test-log_rank0.log $OUTPUT_DIR/$NAME