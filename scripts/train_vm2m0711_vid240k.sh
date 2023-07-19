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

python -m tools.main --config output/VideoMatte240K/vm2m0711_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4_continue/config.yaml --dist --gpus 8 --override \
    train.resume output/VideoMatte240K/vm2m0711_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4_continue wandb.id rduapn00

# python -m tools.main --config configs/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --dist --gpus 4 name debug train.val_iter 10 train.log_iter 5 test.log_iter 5

# python -m tools.main --config configs/VideoMatte240K/vm2m0711_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --dist --gpus 4 --override \
#     name debug \
#     model.weights output/VideoMatte240K/vm2m0711_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
#     train.optimizer.lr 0.00005 \
#     train.max_iter 60000 train.batch_size 2 wandb.use False train.log_iter 1