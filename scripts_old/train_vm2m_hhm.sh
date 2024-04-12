# Update wandb
pip install -U wandb albumentations

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/scripts

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh prepare_hhm.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR
cd ..
python -m tools.main --config configs/HHM/vm2m_short-768-512x512_bs16_100k_adamw_2e-4.yaml --override --dist --gpus 8