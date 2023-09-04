# Update wandb
pip install -U wandb albumentations imgaug scikit-image scipy

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_him_syn.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR

CONFIG=configs/HIM/mgm_enc-dec_multi-inst_atten-loss_him_bs12_0903.yaml
torchrun --nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
            tools/main_ddp.py --config $CONFIG --precision 16 name ours_${WORLD_SIZE}_gpus-${RUNAI_NUM_OF_GPUS}_no-mask_enc_no-spconv-dec_0903 output_dir output/HIM/ablation_study \
            model.decoder res_shortcut_attention_decoder_22 \
            model.backbone res_embed_shortcut_encoder_29 \
            model.backbone_args.num_embed 0 \
            model.backbone_args.num_mask 0