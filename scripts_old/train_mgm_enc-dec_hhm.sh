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
python -m tools.main --config configs/HHM/mgm_enc-dec_hhm_short-768-512x512_bs32_50k_adamw_2e-4.yaml --override --dist --gpus 4 --precision 16 \
        name mgm_enc-dec_ft-hhm_short-768-512x512_bs48_50k_adamw_5.0e-5 \
        dataset.train.short_size 768 \
        dataset.test.short_size 768
