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

python -m tools.main --config configs/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --dist \
                    train.resume output/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4 wandb.id 4baupkau

# python -m tools.main --config configs/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --dist --gpus 4 name debug train.val_iter 10 train.log_iter 5 test.log_iter 5