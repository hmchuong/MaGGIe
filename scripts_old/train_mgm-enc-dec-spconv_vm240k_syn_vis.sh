# Update wandb and some other packages
pip install -U wandb albumentations imgaug scikit-image scipy

ROOT_DIR=/sensei-fs/users/chuongh/vm2m

# Prepare dataset
echo "Preparing dataset..."
cd $ROOT_DIR
cd /mnt/localssd
rsync -av /sensei-fs/users/chuongh/data/vm2m/VideoMatte240K_syn.zip .
unzip -q VideoMatte240K_syn.zip
rsync -av /sensei-fs/users/chuongh/data/vm2m/VIS.zip .
unzip -q VIS.zip

# Start training
echo "Starting training..."
cd $ROOT_DIR

python -m tools.main --config configs/VideoMatte240K/mgm_enc-dec-spconv-temp_vm240ksyn_vis.yaml \
                    --precision 16 --dist --gpus 8