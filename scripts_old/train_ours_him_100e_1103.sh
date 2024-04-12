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
NAME=ours_1105_single-stage_strong-aug_random-crop

if [ -n "$WORLD_SIZE" ] && [ "$WORLD_SIZE" -gt 1 ]; then
    PYCMD="--nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
else
    PYCMD="--standalone --nproc_per_node=$RUNAI_NUM_OF_GPUS"
fi

torchrun $PYCMD tools/main_ddp.py \
                    --config $CONFIG --precision 16 name $NAME wandb.id ofx4ahgp
