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
# python -m tools.main --config configs/HIM/mgm_enc-dec_multi-inst_atten-loss_him_bs48_4gpus_scratch_0820.yaml --dist --gpus 8 --precision 16
CONFIG=configs/HIM/mgm_enc-dec_multi-inst_atten-loss_him_bs12_0903.yaml
torchrun --nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tools/main_ddp.py \
                    --config $CONFIG --precision 16 name ours_short-576-512x512_bs12_nnodes-${WORLD_SIZE}_gpus-${RUNAI_NUM_OF_GPUS}_20k_adamw_1.5e-4_0903