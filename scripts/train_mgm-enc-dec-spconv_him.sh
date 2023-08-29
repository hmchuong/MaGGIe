# Update wandb
pip install -U wandb albumentations imgaug scikit-image scipy

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_him.sh
cd /mnt/localssd
rsync -av /sensei-fs/users/chuongh/data/vm2m/BG20K/train.tar .
if [ ! -d "train" ]; then
    tar -xf train.tar
    mv train bg
fi
if [ ! -d "HHM" ]; then
    mkdir HHM
    cd HHM
    rsync -av /sensei-fs/users/chuongh/data/vm2m/hhm_synthesized.tar.gz .
    tar -xf hhm_synthesized.tar.gz
fi
# Start training
echo "Starting training..."
cd $ROOT_DIR
# python -m tools.main --config configs/HIM/mgm_enc-dec_multi-inst_atten-loss_him_bs48_4gpus_scratch_0820.yaml --dist --gpus 8 --precision 16
CONFIG=configs/HIM/mgm_enc-dec_multi-inst_atten-loss_him_bs12_scratch_0828.yaml
torchrun --nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tools/main_ddp.py --config $CONFIG --precision 16 name mgm_enc-dec-spconv_him-syn-maskrcnn_short-576-512x512_bs12_nnodes-${WORLD_SIZE}_gpus-${RUNAI_NUM_OF_GPUS}_20k_adamw_1.5e-4_scratch_0828