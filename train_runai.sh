# Update wandb
pip install -U wandb

# Prepare dataset
echo "Preparing dataset..."
DATASET=/mnt/localssd/VideoMatte240K
if [ ! -d "$DATASET" ]; then
    rsync -av /sensei-fs/users/chuongh/VideoMatte240K.tar.gz /mnt/localssd/
    rsync -av /sensei-fs/users/chuongh/bg.tar.gz /mnt/localssd/
    cd /mnt/localssd/
    tar -xf VideoMatte240K.tar.gz
    tar -xf bg.tar.gz
fi

# Start training
echo "Starting training..."
cd /sensei-fs/users/chuongh/vm2m
python -m tools.main --config configs/vm2m_baseline_short-768_a100.yaml --override --dist name baseline_rn34_0614 wandb.use True
