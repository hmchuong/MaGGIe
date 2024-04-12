# Update wandb
pip install -U wandb==0.15.0 albumentations==1.3.0 imgaug==0.4.0 scikit-image==0.20.0 scipy==1.10.1

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_hhm.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR
python -m tools.main --config configs/HHM/mgmdk_8x8_hhm_short-768-512x512_bs32_50k_adamw_2e-4.yaml --override --dist --gpus 8