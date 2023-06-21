python -m tools.main --config configs/vm2m_baseline_short-768.yaml --dist name baseline_rn34_0613 train.resume output/baseline_rn34_0613 wandb.id hqy6f07j

# python -m tools.main --config configs/vm2m_baseline_short-768_docker.yaml --dist --override name debug wandb.use True

# python -m tools.main --config configs/vm2m_baseline_short-768.yaml name baseline_rn34_0613 train.resume output/baseline_rn34_0613 wandb.use False