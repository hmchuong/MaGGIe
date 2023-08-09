# Update wandb
pip install -U wandb albumentations imgaug scikit-image scipy

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_him.sh
cd /mnt/localssd
rsync -av /sensei-fs/users/chuongh/data/vm2m/BG20K/train.tar .
tar -xf train.tar
mv train bg
mkdir HHM
cd HHM
rsync -av /sensei-fs/users/chuongh/data/vm2m/hhm_synthesized.tar.gz .
tar -xf hhm_synthesized.tar.gz

# Start training
echo "Starting training..."
cd $ROOT_DIR
python -m tools.main --config output/HIM/mgm_enc-dec-spconv_him_syn_short-576-512x512_bs48_20k_adamw_5.0e-5/config.yaml --override --dist --gpus 4 \
    name mgm_enc-dec-spconv_him_syn_short-576-512x512_bs48_20k_adamw_5.0e-5 \
    train.resume output/HIM/mgm_enc-dec-spconv_him_syn_short-576-512x512_bs48_20k_adamw_5.0e-5 \
    wandb.id 0tzyk097