import os
import argparse
import logging
import torch
import numpy as np
from yacs.config import _assert_with_logging, _check_and_coerce_cfg_value_type

import wandb
import torch.distributed as dist
import torch.multiprocessing as mp

from vm2m.utils import CONFIG
from vm2m.engine import train
from vm2m.engine.test_instmatt import test

def main(rank, cfg, dist_url=None, world_size=8, eval_only=False, precision=32, is_sweep=False, instmatt_dir=None):

    # Set up logger
    logFormatter = logging.Formatter("%(asctime)s [rank " + str(rank) + "] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/{1}log_rank{2}.log".format(cfg.output_dir, "test-" if eval_only else "", str(rank)))
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
            # wandb.login(host=os.getenv("WANDB_BASE_URL"), key=os.getenv("WANDB_API_KEY"), force=True)
            if not is_sweep:
                if cfg.wandb.id == '':
                    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.name)
                else:
                    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, 
                            name=cfg.name, id=cfg.wandb.id, resume='must')
            wandb.config.update(cfg, allow_val_change=True)
    if eval_only:
        test(cfg, rank, dist_url is not None, matte_dir=instmatt_dir)
    else:
        train(cfg, rank, dist_url is not None, precision)
    
    if dist_url is not None:
        dist.destroy_process_group()

def merge_from_list(config, cfg_list):
    """Merge config (keys, values) in a list (e.g., from command line) into
    this CfgNode. For example, `cfg_list = ['FOO.BAR', 0.5]`.
    """
    _assert_with_logging(
        len(cfg_list) % 2 == 0,
        "Override list has odd length: {}; it must be a list of pairs".format(
            cfg_list
        ),
    )
    root = config
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if root.key_is_deprecated(full_key):
            continue
        if root.key_is_renamed(full_key):
            root.raise_key_rename_error(full_key)
        key_list = full_key.split(".")
        d = config
        for subkey in key_list[:-1]:
            _assert_with_logging(
                subkey in d, "Non-existent key: {}".format(full_key)
            )
            d = d[subkey]
        subkey = key_list[-1]
        _assert_with_logging(subkey in d, "Non-existent key: {}".format(full_key))
        value = config._decode_cfg_value(v)
        if type(value) != type(d[subkey]):
            value = type(d[subkey])(value)
        value = _check_and_coerce_cfg_value_type(value, d[subkey], subkey, full_key)
        d[subkey] = value

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Video Mask To Matte")
    parser.add_argument("--config", type=str, default="configs/config.toml", help="Path to config file")
    parser.add_argument("--override", action="store_true", help="Override the experiment")
    parser.add_argument("--dist", action="store_true", help="Use distributed training")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs for distributed training")
    parser.add_argument("--precision", type=int, default=32, help="Precision for distributed training")
    parser.add_argument("--dist-url", type=str, default="", help="Distributed training URL")
    parser.add_argument("--sweep-job", action='store_true', help="Whether this is a sweep job")
    parser.add_argument("--instmatt-dir", type=str, default="", help="InstMatte prediction directory")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate the model")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='modify config file from terminal')
    args = parser.parse_args()

    CONFIG.merge_from_file(args.config)
    new_opts = []
    for opt in args.opts:
        if opt.startswith('--'):
            new_opts.append(opt[2:])
        elif '=' in opt:
            key, value = opt.split('=')
            new_opts.append(key)
            new_opts.append(value)
        else:
            new_opts.append(opt)
    merge_from_list(CONFIG, new_opts)

    # Random seed in case seed is not provided in config file.
    seed = CONFIG.train.seed
    if seed == -1:
        seed = np.random.randint(1, 10000)
    CONFIG.train.seed = seed
    
    # Dump config to the file
    output_dir = os.path.join(CONFIG.output_dir, CONFIG.name)

    os.makedirs(output_dir, exist_ok=True)
    if not args.eval_only:    
        with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
            f.write(CONFIG.dump())

    CONFIG.output_dir = output_dir
    dist_url = args.dist_url
    if dist_url == '':
        dist_url = "tcp://127.0.0.1:{}".format(np.random.randint(1200, 1300))

    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    os.environ["WANDB_START_METHOD"] = "thread"
    if args.dist:
        mp.spawn(main, nprocs=args.gpus, args=(CONFIG, dist_url, args.gpus, args.eval_only, args.precision, args.sweep_job, args.instmatt_dir))
    else:
        main(0, CONFIG, None, 1, args.eval_only, args.precision, args.sweep_job, args.instmatt_dir)