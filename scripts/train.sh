# python -m tools.main --config configs/vm2m_baseline_short-768.yaml --dist --override --gpus 4 name debug

# python -m tools.main --config configs/vm2m_baseline_short-768_docker.yaml --dist --override name debug wandb.use True

# python -m tools.main --config configs/vm2m_baseline_short-768.yaml name baseline_rn34_0613 train.resume output/baseline_rn34_0613 wandb.use False

# python -m tools.main --config configs/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b32-f8_100k_adamw_1e-4.yaml --override --dist --gpus 4 name debug \
#         train.batch_size 2

DEBUG=1 python -m tools.main --config configs/VideoMatte240K/mgm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --override --gpus 8 name debug wandb.use True train.log_iter 10

DEBUG=1 python -m tools.main --config configs/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --override --gpus 4 name debug wandb.use False train.log_iter 1

DEBUG=1 python -m tools.main --config configs/VideoMatte240K/vm2m0711_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --override --gpus 4 name debug wandb.use False train.log_iter 1 train.batch_size 2

python -m tools.main --config configs/VideoMatte240K/mgm_sftm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --dist --gpus 4 name debug wandb.use True train.log_iter 1 train.vis_iter 25 train.val_iter 50 train.batch_size 2

python -m tools.main --config configs/HIM/mgm_him_short-768-512x512_bs32_50k_adamw_2e-4.yaml --gpus 4 name debug wandb.use False train.log_iter 1 train.vis_iter 25 train.val_iter 50

python -m tools.main --config configs/HIM/mgm_him_short-768-512x512_bs32_50k_adamw_2e-4.yaml --dist --gpus 4