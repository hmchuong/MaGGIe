# Update wandb
pip install -U wandb albumentations imgaug scikit-image scipy

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_vhm_syn_real.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR
CONFIG=configs/VideoMatte240K/ours_ss_1022.yaml
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
torchrun --nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tools/main_ddp.py \
            --config $CONFIG --precision 16 name ours_ss_1023 train.self_train.end_ratio 0.1 wandb.id d5lbfjjd train.resume_last True

# torchrun --standalone --nnodes=1 --nproc_per_node=1 tools/main_ddp.py --config $CONFIG --precision 16 name debug wandb.use False train.resume_last False