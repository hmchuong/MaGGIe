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
CONFIG=configs/HIM/mgm_cvpr23_multi-inst_him_bs32.yaml
torchrun --standalone --nnodes=1 --nproc_per_node=$RUNAI_NUM_OF_GPUS tools/main_ddp.py \
                    --config $CONFIG --precision 16