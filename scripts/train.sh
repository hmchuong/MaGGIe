# python -m tools.main --config configs/vm2m_baseline_short-768.yaml --dist --override --gpus 4 name debug

# python -m tools.main --config configs/vm2m_baseline_short-768_docker.yaml --dist --override name debug wandb.use True

# python -m tools.main --config configs/vm2m_baseline_short-768.yaml name baseline_rn34_0613 train.resume output/baseline_rn34_0613 wandb.use False

python -m tools.main --config configs/HHM/vm2m_short-768-512x512_bs16_100k_adamw_2e-4.yaml --override --gpus 4 name debug train.val_iter 100 train.log_iter 10 train.vis_iter 50 train.num_workers 0