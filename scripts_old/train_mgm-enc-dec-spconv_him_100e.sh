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
# python -m tools.main --config configs/HIM/mgm_enc-dec_multi-inst_atten-loss_him_bs48_4gpus_scratch_0820.yaml --dist --gpus 8 --precision 16
CONFIG=configs/HIM/mgm_enc-dec_multi-inst_atten-loss_him_bs12_0903.yaml
torchrun --standalone --nnodes=1 --nproc_per_node=$RUNAI_NUM_OF_GPUS  tools/main_ddp.py \
                    --config $CONFIG --precision 16 name ours_fea-dropout_dropout-mask-0.2_0925 \
                    train.max_iter 52000