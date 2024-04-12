# Update wandb and some other packages
pip install -U wandb albumentations imgaug scikit-image scipy

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_hhm.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR

python -m tools.main --config configs/HHM/mgm_swint_atten-dec_hhm_short-768-512x512_bs32_50k_adamw_2e-4.yaml --dist --override --gpus 8

# python -m tools.main --config configs/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --dist --gpus 4 name debug train.val_iter 10 train.log_iter 5 test.log_iter 5

# python -m tools.main --config configs/HHM/mgm_swint_atten-dec_q-16_hhm_short-768-512x512_bs30_50k_adamw_2e-4.yaml --dist --override --gpus 4 name mgm_swint_atten-dec_q-16_hhm_short-768-512x512_bs10_50k_adamw_2e-4 train.batch_size 10