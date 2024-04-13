# torchrun --standalone --nproc_per_node=2 tools/main.py  --config configs/maggie_image.yaml --precision 16 \
#     name test model.weights '' train.val_iter 50 wandb.entity chuonghm train.batch_size 8 dataset.train.root_dir /mnt/disks/data train.num_workers 2

torchrun --standalone --nproc_per_node=2 tools/main.py  --config configs/maggie_video.yaml --precision 16 \
    name test model.weights checkpoints/image/best_model.pth train.val_iter 50 wandb.entity chuonghm train.batch_size 2 \
    dataset.train.root_dir /mnt/disks/data train.num_workers 2 dataset.train.clip_length 4