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

CONFIG=configs/HIM/ours_1102_singlestage_ft.yaml
NAME=ours_1102_single-stage_strong-aug_ft-reweight

if [ -n "$WORLD_SIZE" ] && [ "$WORLD_SIZE" -gt 1 ]; then
    PYCMD="--nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
else
    PYCMD="--standalone --nproc_per_node=$RUNAI_NUM_OF_GPUS"
fi

torchrun $PYCMD tools/main_ddp.py \
                    --config $CONFIG --precision 16 name $NAME

# CONFIG=configs/HIM/ours_1031_ft_details.yaml
# NAME=ours_1031_single-stage_ft-details
# torchrun --standalone --nproc_per_node=8  tools/main_ddp.py \
#                     --config $CONFIG --precision 16 name $NAME
# CONFIG=configs/HIM/ours_1029_multistage_stage2.yaml
# NAME=ours_1029_multi-stage_inst-spec-feat_weightos41_loss-w_stage2
# torchrun --standalone --nproc_per_node=8  tools/main_ddp.py \
#                     --config $CONFIG --precision 16 name $NAME model.decoder_args.detail_mask_dropout 0.0 #wandb.id 5bgfjzxz

# torchrun --standalone --nproc_per_node=$RUNAI_NUM_OF_GPUS  tools/main_ddp.py \
#                     --config $CONFIG --precision 16 name $NAME

# torchrun --nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT  tools/main_ddp.py \
#                     --config $CONFIG --precision 16 name $NAME
