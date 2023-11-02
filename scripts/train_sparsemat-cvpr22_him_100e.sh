# Update wandb
pip install -U wandb albumentations imgaug scikit-image scipy

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_him_syn.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR
# CONFIG=configs/HIM/sparsemat_cvpr22_him_bs64.yaml
CONFIG=configs/HIM/sparsemat_cvpr22_him_bs16.yaml
# torchrun --nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tools/main_ddp.py \
#                     --config $CONFIG --precision 16

torchrun --standalone --nproc_per_node=$RUNAI_NUM_OF_GPUS tools/main_ddp.py \
                    --config $CONFIG --precision 16