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
CONFIG=configs/VideoMatte240K/ours_vhm_ss_0927.yaml
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=8 tools/main_ddp.py --config $CONFIG --precision 16

# torchrun --standalone --nnodes=1 --nproc_per_node=1 tools/main_ddp.py --config $CONFIG --precision 16 name debug wandb.use False train.resume_last False