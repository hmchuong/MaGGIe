# Update wandb
pip install -U wandb

ROOT_DIR=/sensei-fs/users/chuongh/vm2m

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh prepare_vid240k.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR
python -m tools.main --config configs/vm2m_baseline_short-768_a100.yaml --override --dist name baseline_rn34_0614 wandb.use True