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

python -m tools.main --config configs/VideoMatte240K/mgm_m-1_atten-dec_vid240_pre-hmm_s-768-512x512_b6-f8_100k_adamw_1e-4.yaml --dist --gpus 4 name debug wandb.use False train.log_iter 1 train.vis_iter 25 train.val_iter 50 train.batch_size 2


python -m tools.main --config configs/HHM/mgm_atten-dec_q-16_hhm_short-768-512x512_bs30_50k_adamw_2e-4.yaml --gpus 4 name debug wandb.use False \
                                train.log_iter 1 train.vis_iter 25 train.val_iter 50 train.batch_size 15 \
                                model.weights output/HHM/mgm_atten-dec_hhm_short-768-512x512_bs32_50k_adamw_2e-4/best_model.pth

python -m tools.main --config configs/HHM/mgm_enc-atten_dec-atten_hhm_short-768-512x512_bs32_50k_adamw_2e-4.yaml --gpus 4 name debug wandb.use False \
                                train.log_iter 1 train.vis_iter 25 train.val_iter 50 train.batch_size 15

python -m tools.main --config configs/HIM/mgm_enc-embed_dec-embed-atten_him_short-768-512x512_bs8_50k_adamw_2e-4.yaml --dist --gpus 2 --precision 16 \
                    name mgm_enc-dec_him_bs8_3gpus_best0804 \
                    dataset.train.max_inst=5 \
                    dataset.train.use_single_instance_only=True \
                    model.backbone_args.num_embed=3 model.decoder_args.head_channel=50 \
                    model.loss_alpha_grad_w=0.25 model.loss_multi_inst_type=l1 \
                    model.loss_multi_inst_w=0 model.loss_multi_inst_warmup=4000 \
                    model.mgm.warmup_iter=2000 train.optimizer.lr=0.0001 train.scheduler.warmup_iters=1500 \
                    train.max_iter=30000 train.num_workers=4 train.batch_size=12 wandb.use True