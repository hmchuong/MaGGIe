# Update wandb
pip install -U wandb albumentations imgaug scikit-image scipy
pip install -U yapf

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_vhm_syn.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR

CONFIG=configs/VideoMatte240K/ours_vhm_1103.yaml
NAME=ours_vhm_bi-temp_1110_3

if [ -n "$WORLD_SIZE" ] && [ "$WORLD_SIZE" -gt 1 ]; then
    PYCMD="--nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
else
    PYCMD="--standalone --nproc_per_node=$RUNAI_NUM_OF_GPUS"
fi

TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun $PYCMD tools/main_ddp.py \
                    --config $CONFIG --precision 16 name $NAME train.batch_size 4 \
                        train.resume_last True dataset.train.clip_length 5 \
                        model.weights output/HIM/ours_1108_stronger-aug_guidance_scratch/last_model_24.5k.pth # wandb.id s6hvorcn
