# Update wandb
pip install -U wandb albumentations imgaug scikit-image scipy

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_hhm.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR
python -m tools.main --config configs/HHM/mgm_short-768-512x512_bs32_50k_adamw_2e-4.yaml --override --dist --gpus 8 #name debug train.log_iter 10