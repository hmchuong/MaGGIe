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

CONFIG=configs/HIM/mgm_lap_multi-inst_atten-loss_him_bs12_0929.yaml
torchrun --standalone --nnodes=1 --nproc_per_node=$RUNAI_NUM_OF_GPUS  tools/main_ddp.py \
                    --config $CONFIG --precision 16 name ours_fea-dropout-0.2_lap_lap-loss_0930 \
                    train.max_iter 20000