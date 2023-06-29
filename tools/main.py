import os
import argparse
import logging
import torch
import numpy as np

import wandb
import torch.distributed as dist
import torch.multiprocessing as mp

from vm2m.utils import CONFIG
from vm2m.engine import train, test

def main(rank, cfg, dist_url=None, world_size=8, eval_only=False):

    # Set up logger
    logFormatter = logging.Formatter("%(asctime)s [rank " + str(rank) + "] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/log_rank{1}.log".format(cfg.output_dir, str(rank)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    if os.getenv("DEBUG", False):
        rootLogger.setLevel('DEBUG')
    else:
        rootLogger.setLevel('INFO')

    if rank == 0 or os.getenv("DEBUG", False):
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
    
    logging.info("Config:\n" + str(cfg))

    # Set up distributed training
    if dist_url is not None:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", init_method=dist_url, world_size=world_size, rank=rank)

    # Set up wandb
    if rank == 0 and not eval_only:
        if cfg.wandb.use:
            # try:
            #     wandb.login(host=os.getenv("WANDB_BASE_URL"), key=os.getenv("WANDB_API_KEY"))
            # except:
            wandb.login(host=os.getenv("WANDB_BASE_URL"), key=os.getenv("WANDB_API_KEY"), force=True)
            if cfg.wandb.id == '':
                wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.name)
            else:
                wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, 
                           name=cfg.name, id=cfg.wandb.id, resume='must')
            wandb.config.update(cfg, allow_val_change=True)
    if eval_only:
        test(cfg, rank, dist_url is not None)
    else:
        train(cfg, rank, dist_url is not None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Video Mask To Matte")
    parser.add_argument("--config", type=str, default="configs/config.toml", help="Path to config file")
    parser.add_argument("--override", action="store_true", help="Override the experiment")
    parser.add_argument("--dist", action="store_true", help="Use distributed training")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs for distributed training")
    parser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:23456", help="Distributed training URL")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate the model")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='modify config file from terminal')
    args = parser.parse_args()

    CONFIG.merge_from_file(args.config)
    CONFIG.merge_from_list(args.opts)

    # Random seed in case seed is not provided in config file.
    seed = CONFIG.train.seed
    if seed == -1:
        seed = np.random.randint(1, 10000)
    CONFIG.train.seed = seed

    # Dump config to the file
    if not args.eval_only:
        os.makedirs(CONFIG.output_dir, exist_ok=True)
        with open(os.path.join(CONFIG.output_dir, "config.yaml"), 'w') as f:
            f.write(CONFIG.dump())
            
    CONFIG.output_dir = os.path.join(CONFIG.output_dir, CONFIG.name)
    
    # Check output directory
    if (os.path.exists(CONFIG.output_dir) and os.listdir(CONFIG.output_dir) and not args.eval_only and not args.override and not (CONFIG.output_dir == CONFIG.train.resume)):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(CONFIG.output_dir))
    
    

    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    os.environ["WANDB_START_METHOD"] = "thread"

    if args.dist:
        mp.spawn(main, nprocs=args.gpus, args=(CONFIG, args.dist_url, args.gpus, args.eval_only))
    else:
        main(0, CONFIG, None, 1, args.eval_only)
