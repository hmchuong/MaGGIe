# Update wandb and some other packages
pip install -U wandb albumentations imgaug scikit-image scipy

ROOT_DIR=/sensei-fs/users/chuongh/vm2m

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_vid240k.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR

python -m tools.main --config configs/VideoMatte240K/mgm_enc-dec-spconv-temp_os16_os4.yaml \
                    --precision 16 --dist --gpus 8