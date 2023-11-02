# Update wandb and some other packages
pip install -U wandb albumentations imgaug scikit-image scipy

ROOT_DIR=/sensei-fs/users/chuongh/vm2m

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_vhm_syn.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR

CONFIG=configs/VideoMatte240K/sparsemat_vhm_1031.yaml
NAME=sparsemat_hr_1101
torchrun --standalone --nproc_per_node=$RUNAI_NUM_OF_GPUS tools/main_ddp.py \
                    --config $CONFIG --precision 16 train.max_iter 26000 name $NAME
# torchrun --nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tools/main_ddp.py \
#                     --config $CONFIG --precision 16

# python -m tools.main --config configs/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --dist --gpus 4 name debug train.val_iter 10 train.log_iter 5 test.log_iter 5