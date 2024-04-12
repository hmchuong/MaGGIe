# Update wandb
pip install -U wandb albumentations imgaug scikit-image scipy
pip install -U yapf

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_him_syn.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR

CONFIG=configs/HIM/ours_1102_singlestage.yaml
NAME=ours_1106_single-stage_stronger-aug_ft_guidance8 #_random-crop

if [ -n "$WORLD_SIZE" ] && [ "$WORLD_SIZE" -gt 1 ]; then
    PYCMD="--nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
else
    PYCMD="--standalone --nproc_per_node=$RUNAI_NUM_OF_GPUS"
fi

torchrun $PYCMD tools/main_ddp.py \
                    --config $CONFIG --precision 16 name $NAME \
                    model.weights output/HIM/ours_1106_single-stage_stronger-aug_ft_guidance2/best_model.pth \
                    model.mgm.warmup_iter 0 model.decoder_args.warmup_detail_iter 0 \
                    model.freeze_coarse False train.optimizer.lr 0.00002 train.batch_size 12 \
                    model.loss_multi_inst_w 1.0 model.loss_multi_inst_warmup 0

# torchrun $PYCMD tools/main_ddp.py \
#                     --config $CONFIG --precision 16 name $NAME \
#                     model.weights output/HIM/ours_1106_stronger-aug_from-scratch/best_model.pth \
#                     model.mgm.warmup_iter 0 model.decoder_args.warmup_detail_iter 0 model.freeze_coarse False train.optimizer.lr 0.00005 train.batch_size 12

