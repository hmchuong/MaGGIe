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

CONFIG=configs/HIM/ours_1025.yaml
NAME=ours_1029_hr-detail
# torchrun --standalone --nproc_per_node=$RUNAI_NUM_OF_GPUS  tools/main_ddp.py \
#                     --config $CONFIG --precision 16 name $NAME

torchrun --nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT  tools/main_ddp.py \
                    --config $CONFIG --precision 16 name $NAME train.batch_size 12 train.max_iter 26000
