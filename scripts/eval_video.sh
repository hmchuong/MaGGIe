OUTPUT_DIR=output/video
NAME=eval_full_$2
for SUBSET in easy medium hard
do
    torchrun --standalone --nproc_per_node=1 tools/main.py --config $1 --eval-only \
                        name $NAME \
                        output_dir $OUTPUT_DIR \
                        dataset.test.split comp_$SUBSET \
                        test.save_results True \
                        test.log_iter 10
done