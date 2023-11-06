# Update wandb
pip install -U wandb albumentations imgaug scikit-image scipy
pip install -U yapf

ROOT_DIR=/sensei-fs/users/chuongh/vm2m/

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
sh scripts/prepare_him_syn.sh

# Start training
echo "Starting training..."
cd $ROOT_DIR

CONFIG=configs/HIM/ours_1102_image_loss_abl.yaml
NAME=ours_1103_abl_$POSTFIX
# ATTEN_W=5.0
# INST_W=5.0
# CUSTOM_WEIGHT=true

if [ -n "$WORLD_SIZE" ] && [ "$WORLD_SIZE" -gt 1 ]; then
    PYCMD="--nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
else
    PYCMD="--standalone --nproc_per_node=$RUNAI_NUM_OF_GPUS"
fi

torchrun $PYCMD tools/main_ddp.py \
                    --config $CONFIG --precision 16 name $NAME model.loss_atten_w $ATTEN_W model.loss_multi_inst_w $INST_W model.reweight_os8 $CUSTOM_WEIGHT output_dir output/HIM/ablation_loss wandb.id f60hl3uj train.resume_last True