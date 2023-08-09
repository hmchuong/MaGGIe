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
python -m tools.main --config configs/HIM/mgm_enc-embed_dec-embed-atten_him_short-768-512x512_bs8_50k_adamw_2e-4.yaml --dist --gpus 4 \
                    name mgm_enc-dec_him_bs12_4gpus_best0804 \
                    dataset.train.max_inst=5 \
                    dataset.train.use_single_instance_only=True \
                    model.backbone_args.num_embed=3 model.decoder_args.head_channel=50 \
                    model.loss_alpha_grad_w=0.25 model.loss_multi_inst_type=l1 \
                    model.loss_multi_inst_w=0 model.loss_multi_inst_warmup=4000 \
                    model.mgm.warmup_iter=1000 train.optimizer.lr=0.0001 train.scheduler.warmup_iters=750 \
                    train.max_iter=30000 train.num_workers=8

python -m tools.main --config configs/HIM/mgm_enc-embed_dec-embed-atten_him_short-768-512x512_bs8_50k_adamw_2e-4.yaml --dist --gpus 2 --precision 16 dataset.train.max_inst=5 dataset.train.use_single_instance_only=True model.backbone_args.num_embed=3 model.decoder_args.head_channel=50 model.loss_alpha_grad_w=0.25 model.loss_multi_inst_type=smooth_l1_0.5 model.loss_multi_inst_w=0.25 
model.loss_multi_inst_warmup=4000 model.mgm.warmup_iter=3000 train.optimizer.lr=0.00015 train.scheduler.warmup_iters=1000 name mgm_enc-dec_multi-inst_him_bs12_2gpus_best0804 train.max_iter=30000 wandb.id dylgrew5 train.resume output/HIM/mgm_enc-dec_multi-inst_him_bs12_2gpus_best0804