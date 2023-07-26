# Update wandb and some other packages
pip install -U wandb==0.15.0 albumentations==1.3.0 imgaug==0.4.0 scikit-image==0.20.0 scipy==1.10.1

ROOT_DIR=/sensei-fs/users/chuongh/vm2m

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_vid240k.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR

python -m tools.main --config configs/VideoMatte240K/mgm_m-1_atten-dec_vid240_pre-hmm_s-768-512x512_b6-f8_100k_adamw_1e-4.yaml --dist --gpus 4