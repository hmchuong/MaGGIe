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
python -m tools.main --config configs/HIM/mgm_enc-dec_him_syn_short-576-512x512_bs48_20k_adamw_5.0e-5.yaml --override --dist --gpus 8 \
                        name mgm_enc-dec_topkrec_him_syn_short-576-512x512_bs256_20k_adamw_5.0e-5 \
                        train.max_iter 20000 \
                        train.batch_size 32 \
                        train.scheduler.warmup_iters 500 \
                        model.mgm.warmup_iter 5000