# Update wandb
pip install -U wandb albumentations imgaug scikit-image scipy

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_vhm_syn.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR
CONFIG=configs/VideoMatte240K/ours_vhm_1010.yaml
NAME=ours_vhm_transition_pretrained_1010
# TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tools/main_ddp.py \
#                     --config $CONFIG --precision 16 model.weights output/VHM/ours_vhm_0905/best_model.pth

# TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=$RUNAI_NUM_OF_GPUS tools/main_ddp.py \
#                     --config $CONFIG --precision 16 name $NAME
# export DEBUG=1
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tools/main_ddp.py \
                    --config $CONFIG --precision 16 name $NAME train.batch_size 8