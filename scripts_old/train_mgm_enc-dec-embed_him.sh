# Update wandb
pip install -U wandb albumentations imgaug scikit-image scipy

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_hhm.sh
sh scripts/prepare_him.sh
cd /mnt/localssd
rsync -av /sensei-fs/users/chuongh/data/vm2m/BG20K/train.tar .
tar -xf train.tar
mv train bg

# Start training
echo "Starting training..."
cd $ROOT_DIR
python -m tools.main --config configs/HIM/mgm_enc-embed_dec-embed-atten_him_short-768-512x512_bs8_50k_adamw_2e-4.yaml --override --dist --gpus 4