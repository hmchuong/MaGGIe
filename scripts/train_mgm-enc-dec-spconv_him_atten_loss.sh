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
python -m tools.main --config configs/HIM/mgm_enc-dec_multi-inst_atten-loss_him_bs48_4gpus_scratch_0820.yaml --dist --gpus 8 --precision 16 \
                                name mgm_enc-dec-spconv_him-syn-maskrcnn_short-576-512x512_bs48_30k_adamw_1.5e-4_scratch_0821 \
                                model.decoder_args.warmup_mask_atten_iter 25000