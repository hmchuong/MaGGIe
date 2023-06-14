python -m tools.main --config configs/vm2m_baseline_short-768.yaml --dist name baseline_rn34_0613 train.resume output/baseline_rn34_0613 wandb.id hqy6f07j

# python -m tools.train --config configs/vm2m_baseline.yaml --override name debug wandb.use False

# python -m tools.main --config configs/vm2m_baseline_short-768.yaml name baseline_rn34_0613 train.resume output/baseline_rn34_0613 wandb.use False