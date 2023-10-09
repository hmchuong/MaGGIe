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
CONFIG=configs/VideoMatte240K/ours_vhm_1008.yaml
# TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tools/main_ddp.py \
#                     --config $CONFIG --precision 16 model.weights output/VHM/ours_vhm_0905/best_model.pth

# TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=$RUNAI_NUM_OF_GPUS tools/main_ddp.py \
#                     --config $CONFIG --precision 16 name ours_vhm_mem-query-pre_1003

TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tools/main_ddp.py \
                    --config $CONFIG --precision 16 name ours_vhm_mem-query-mlp_1008 model.decoder_args.use_query_temp mlp