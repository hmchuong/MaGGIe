
from functools import partial
import os
import signal
import itertools
import random
import time
import torch
import logging
import numpy as np
import wandb
from torch.utils import data as torch_data
from torch.cuda.amp import autocast, GradScaler
from vm2m.dataloader import MultiInstVidDataset
from vm2m.network import build_model
from vm2m.utils.dist import AverageMeter, reduce_dict
from vm2m.utils.metric import build_metric

from .optim import build_optim_lr_scheduler
from .test import val_video as val
from .train import load_state_dict, wandb_log_image

global batch
batch = None

def create_train_dataset(cfg):
    # create syn dataset and real dataset
    dataset_cfg = cfg.dataset.train
    syn_dataset = MultiInstVidDataset(root_dir=dataset_cfg.root_dir, split=dataset_cfg.split, 
                                            clip_length=dataset_cfg.clip_length, padding_inst=dataset_cfg.padding_inst, is_train=True, short_size=dataset_cfg.short_size, 
                                            crop=dataset_cfg.crop, flip_p=dataset_cfg.flip_prob,
                                            max_step_size=dataset_cfg.max_step_size, random_seed=cfg.train.seed)
    real_dataset = MultiInstVidDataset(root_dir=dataset_cfg.root_dir, split=dataset_cfg.ss_split, 
                                            clip_length=3, padding_inst=dataset_cfg.padding_inst, is_train=True, short_size=dataset_cfg.short_size, 
                                            crop=dataset_cfg.crop, flip_p=dataset_cfg.flip_prob,
                                            max_step_size=dataset_cfg.max_step_size, random_seed=cfg.train.seed, 
                                            pha_dir=dataset_cfg.mask_dir_name, mask_dir_name=dataset_cfg.mask_dir_name, is_ss_dataset=True)
    return syn_dataset, real_dataset

def create_test_dataset(cfg):
    # create test datasets: comp_medium and real
    dataset_cfg = cfg.dataset.test
    syn_dataset = MultiInstVidDataset(root_dir=dataset_cfg.root_dir, split=dataset_cfg.split, 
                                      clip_length=dataset_cfg.clip_length, overlap=dataset_cfg.clip_overlap, is_train=False, short_size=dataset_cfg.short_size, 
                                      random_seed=cfg.train.seed, mask_dir_name=dataset_cfg.mask_dir_name, pha_dir=dataset_cfg.alpha_dir_name)
    real_dataset = MultiInstVidDataset(root_dir=dataset_cfg.root_dir, split="real", 
                                      clip_length=dataset_cfg.clip_length, overlap=dataset_cfg.clip_overlap, is_train=False, short_size=dataset_cfg.short_size, 
                                      random_seed=cfg.train.seed, mask_dir_name=dataset_cfg.mask_dir_name, pha_dir=dataset_cfg.alpha_dir_name)
    return syn_dataset, real_dataset

def evaluation_ss(rank, val_model, is_dist, val_syn_error_dict, val_real_error_dict, val_syn_loader, val_real_loader, cfg, device, epoch, i_cycle, real_ratio, global_step):
    logging.info("Start validation...")
    _ = [v.reset() for v in val_syn_error_dict.values()]
    _ = [v.reset() for v in val_real_error_dict.values()]
    if rank == 0:
        logging.info("Evaluating the synthetic dataset...")
        _ = val(val_model, val_syn_loader, device, cfg.test.log_iter, val_syn_error_dict, do_postprocessing=False, use_trimap=False, callback=None, use_temp=False)

    if rank == 1:
        logging.info("Evaluating the real dataset...")
        _ = val(val_model, val_real_loader, device, cfg.test.log_iter, val_real_error_dict, do_postprocessing=False, use_trimap=False, callback=None, use_temp=False)

    for k, v in val_real_error_dict.items():
        v.gather_metric(0)
    
    if rank == 0:
        log_str = "Validation syn data:"
        for k, v in val_syn_error_dict.items():
            log_str += "{}: {:.4f}, ".format(k, v.average())
        logging.info(log_str)
        log_str = "Validation real data:"
        for k, v in val_real_error_dict.items():
            log_str += "{}: {:.4f}, ".format(k, v.average())
        logging.info(log_str)

    # Log wandb
    if cfg.wandb.use and rank == 0:
        for k, v in val_syn_error_dict.items():
            wandb.log({"val/syn_" + k: v.average()}, commit=False)
        for k, v in val_real_error_dict.items():
            wandb.log({"val/real_" + k: v.average()}, commit=False)
        wandb.log({"val/epoch": epoch, 
                    "val/cycle": i_cycle,
                    "val/real_ratio": real_ratio,
                    "val/global_step": global_step
                    }, commit=True)




def graceful_exit_handler(signum, frame, global_rank):
    global batch
    logging.info(f"Exit {signum} Saving batch...")
    import pickle
    pickle.dump(batch, open(f"batch_{global_rank}.pkl", "wb"))
    exit(1)

def train_ss(cfg, rank, is_dist=False, precision=32, global_rank=None):
    if global_rank is None:
        global_rank = rank
    
    global batch
    
    # signal.signal(signal.SIGTERM, partial(graceful_exit_handler, global_rank=global_rank))
    # signal.signal(signal.SIGINT, partial(graceful_exit_handler, global_rank=global_rank))
    # signal.signal(signal.SIGQUIT, partial(graceful_exit_handler, global_rank=global_rank))
    # signal.signal(signal.SIGABRT, partial(graceful_exit_handler, global_rank=global_rank)) 

    device = f'cuda:{rank}'

    # Create dataset
    logging.info("Creating train dataset...")

    # Create datasets
    train_syn_dataset, train_real_dataset = create_train_dataset(cfg)

    

    # Create dataloader
    if is_dist:
        train_syn_sampler = torch_data.DistributedSampler(train_syn_dataset)
        train_real_sampler = torch_data.DistributedSampler(train_real_dataset)
    else:
        train_syn_sampler = None
        train_real_sampler = None

    
    g = torch.Generator()
    g.manual_seed(cfg.train.seed)

    train_syn_loader = torch_data.DataLoader(
        train_syn_dataset, batch_size=cfg.train.batch_size, shuffle=(train_syn_sampler is None),
        num_workers=cfg.train.num_workers,
        pin_memory=True, sampler=train_syn_sampler,
        generator=g)
    
    train_real_loader = torch_data.DataLoader(
        train_real_dataset, batch_size=cfg.train.batch_size, shuffle=(train_real_sampler is None),
        num_workers=cfg.train.num_workers,
        pin_memory=True, sampler=train_real_sampler,
        generator=g)
    
    # Validate only at rank 0
    # if rank == 0:
    logging.info("Creating val dataset...")
    val_syn_dataset, val_real_dataset = create_test_dataset(cfg)
    
    # TODO: Debug
    # val_syn_dataset.frame_ids = val_syn_dataset.frame_ids[:100]
    # val_real_dataset.frame_ids = val_real_dataset.frame_ids[:100]

    val_syn_loader = torch_data.DataLoader(
        val_syn_dataset, batch_size=cfg.test.batch_size, shuffle=False, pin_memory=True,
        sampler=None,
        num_workers=cfg.test.num_workers)
    val_real_loader = torch_data.DataLoader(
        val_real_dataset, batch_size=cfg.test.batch_size, shuffle=False, pin_memory=True,
        sampler=None,
        num_workers=cfg.test.num_workers)
    
    # Build model
    logging.info("Building model...")
    model = build_model(cfg.model)
    model = model.to(device)
    training_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logging.info("Number of trainable parameters: {}".format(training_params))

    # Define optimizer and lr scheduler
    logging.info("Building optimizer and lr scheduler...")
    # import pdb; pdb.set_trace()
    cfg.train.max_iter = cfg.train.self_train.max_cycles * cfg.train.self_train.epoch_per_cycle * cfg.train.self_train.iter_per_epoch
    optimizer, lr_scheduler = build_optim_lr_scheduler(cfg, model)

    if is_dist:
        if cfg.model.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        having_unused_params = True
        model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[rank], find_unused_parameters=having_unused_params)

    # Load pretrained model
    if os.path.isfile(cfg.model.weights):
        logging.info("Loading pretrained model from {}".format(cfg.model.weights))
        state_dict = torch.load(cfg.model.weights, map_location=device)
        # if is_dist:
        #     missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)
        # else:
        #     missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        missing_keys, unexpected_keys, mismatch_keys = load_state_dict(model if not is_dist else model.module, state_dict)
        if len(missing_keys) > 0:
            logging.warn("Missing keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            logging.warn("Unexpected keys: {}".format(unexpected_keys))
        if len(mismatch_keys) > 0:
            logging.warn("Mismatch keys: {}".format(mismatch_keys))

    # Resume training
    start_epoch = 0
    start_cycle = 0
    n_real_images = 0
    n_real_instances = 0
    n_syn_samples = 0
    n_syn_instances = 0
    if cfg.train.resume_last:
        if os.path.exists(os.path.join(cfg.output_dir, 'last_opt.pth')):
            logging.info("Resuming last model from {}".format(cfg.output_dir))
            # try:
            opt_dict = torch.load(os.path.join(cfg.output_dir, 'last_opt.pth'), map_location=device)
            start_epoch = opt_dict['epoch']
            start_cycle = opt_dict['cycle']
            n_real_images = opt_dict['n_real_samples']
            n_real_instances = opt_dict['n_real_instances']
            n_syn_samples = opt_dict['n_syn_samples']
            n_syn_instances = opt_dict['n_syn_instances']
            # Load optimizer and lr_scheduler
            optimizer.load_state_dict(opt_dict['optimizer'])
            lr_scheduler.load_state_dict(opt_dict['lr_scheduler'])
            logging.info("Done loading optimizer and lr_scheduler")

            model_name = 'model_{}_{}.pth'.format(start_cycle, start_epoch)
            logging.info("Loading model from {}".format(model_name))
            state_dict = torch.load(os.path.join(cfg.output_dir, model_name), map_location=device)
            if is_dist:
                model.module.load_state_dict(state_dict, strict=True)
            else:
                model.load_state_dict(state_dict, strict=True)
            logging.info("Resumed from cycle {}, epoch {}".format( start_cycle, start_epoch))
            start_epoch = start_epoch + 1
            # except:
            #     logging.info("Cannot resume last model from {}".format(cfg.output_dir))


    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    
    log_metrics = {}
    end_time = time.time()

    # Build validation metrics
    val_real_error_dict = build_metric(cfg.train.val_metrics)
    val_syn_error_dict = build_metric(cfg.train.val_metrics)
    assert len(val_real_error_dict) > 0, "No validation metrics found!"

    # Start training
    logging.info("Start training...")
    model.train()
    n_cycles = cfg.train.self_train.max_cycles
    n_epochs = cfg.train.self_train.epoch_per_cycle
    n_iters = cfg.train.self_train.iter_per_epoch
    global_step = 0

    real_ratios = torch.linspace(cfg.train.self_train.start_ratio, cfg.train.self_train.end_ratio, n_epochs)
    scaler = GradScaler() if precision == 16 else None
    syn_epoch = 0
    real_epoch = 0
    
    # Set the epoch
    if is_dist:
        train_syn_sampler.set_epoch(syn_epoch)
        train_real_sampler.set_epoch(real_epoch)

    train_real_iterator = iter(train_real_loader)
    train_syn_iterator = iter(train_syn_loader)
    
    if start_cycle == 0 and start_epoch == 0:
        model.eval()
        val_model = model.module if is_dist else model
        evaluation_ss(global_rank, val_model, is_dist, val_syn_error_dict, val_real_error_dict, val_syn_loader, val_real_loader, cfg, device, 0, 0, 0, 0)

    for i_cycle in range(start_cycle, n_cycles):
        logging.info("Starting cycle: {}".format(i_cycle))
        
        for epoch in range(start_epoch, n_epochs):
            model.train()

            # Recompute the ratio of syn and real across GPUs
            real_iters = int(real_ratios[epoch] * n_iters)
            real_masks = torch.zeros(n_iters, dtype=torch.bool, device=device)
            real_masks[torch.randperm(n_iters)[:real_iters]] = True
            
            logging.info("Ratio of real: {:.4f} ~ {}".format(real_ratios[epoch], real_iters))

            if is_dist:
                torch.distributed.broadcast(real_masks, 0)
            
            for step in range(n_iters):
                try:
                    # Train here
                    global_step = i_cycle * n_epochs * n_iters + epoch * n_iters + step
                    is_real = real_masks[step]
                    batch = None
                    if is_real:
                        try:
                            batch = next(train_real_iterator)
                        except StopIteration:
                            logging.info("End of real dataset, resetting...")
                            if is_dist:
                                real_epoch += 1
                                train_real_sampler.set_epoch(real_epoch)
                            train_real_iterator = iter(train_real_loader)
                            batch = next(train_real_iterator)
                    else:
                        try:
                            batch = next(train_syn_iterator)
                        except StopIteration:
                            logging.info("End of syn dataset, resetting...")
                            if is_dist:
                                syn_epoch += 1
                                train_syn_sampler.set_epoch(syn_epoch)
                            train_syn_iterator = iter(train_syn_loader)
                            batch = next(train_syn_iterator)
                    
                    # For logging number of instances, number of images
                    n_images = batch['image'].shape[0] * batch['image'].shape[1]
                    masks = batch['mask']
                    valid_instances = torch.sum(masks, dim=(-1, -2)) > 0
                    n_instances = valid_instances.sum().item()
                    if is_real:
                        n_real_images += n_images
                        n_real_instances += n_instances
                    else:
                        n_syn_samples += n_images
                        n_syn_instances += n_instances

                    data_time.update(time.time() - end_time)
                    
                    # import pickle
                    # load_batch = pickle.load(open(f'/home/chuongh/vm2m/error_batch_{global_rank}.pkl', 'rb'))
                    # is_real = True
                    # batch = {k: v.to(device) for k, v in load_batch.items() if k != 'iter'}
                    # batch['iter'] = load_batch['iter']

                    batch = {k: v.to(device) for k, v in batch.items()}
                    batch['iter'] = global_step
                    # torch.save(model.module.state_dict(), f"debug_model_{global_rank}.pth")

                    optimizer.zero_grad()
                    try:
                        if precision == 16:
                            with autocast():
                                output, loss = model(batch, mem_feat=None, is_real=is_real)
                        else:
                            output, loss = model(batch, mem_feat=None, is_real=is_real)
                    except ValueError as e:
                        import pickle
                        pickle.dump(batch, open(f"error_batch_{global_rank}.pkl", "wb"))
                        raise e
                    
                    if global_rank == 0 and is_real and cfg.wandb.use:
                        # Visualize to wandb
                        try:
                            wandb_log_image(batch, output, global_step - 1)
                        except:
                            pass

                    # import pickle
                    # pickle.dump(batch, open(f"batch_{global_rank}.pkl", "wb"))
                    # logging.info("Is real: {}".format(is_real))
                    # logging.info("batch size {}".format(batch['mask'].shape))
                    # logging.info("Done forward {}".format(loss['total']))

                    if precision == 16:
                        scaler.scale(loss['total']).backward()
                    else:
                        loss['total'].backward()

                    # logging.info("Done backward")

                    # Clip norm
                    if precision == 16:
                        scaler.unscale_(optimizer)
                    all_params = itertools.chain(*[x["params"] for x in optimizer.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, 0.01)
                    
                    # logging.info("Done clipnorm")

                    # Update
                    if precision == 16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    lr_scheduler.step()
                    batch_time.update(time.time() - end_time)
                    # logging.info("Done optimizing")

                    # Store to log_metrics
                    for k, v in loss.items():
                        if k not in log_metrics:
                            log_metrics[k] = AverageMeter(k)
                        log_metrics[k].update(v.item())

                    # logging.info("Cycle: {}, Epoch: {}, Iter: {}, is_real: {}".format(i_cycle, epoch, step, is_real))
                    # Logging
                    if step % cfg.train.log_iter == 0:
                        log_str = "Cycle: {}, Epoch: {}, Iter: {}, is_real: {}".format(i_cycle, epoch, step, is_real)
                        for k, v in log_metrics.items():
                            log_str += ", {}: {:.4f}".format(k, v.avg)
                        log_str += ", lr: {:.6f}".format(lr_scheduler.get_last_lr()[0])
                        log_str += ", batch_time: {:.4f}s".format(batch_time.avg)
                        log_str += ", data_time: {:.4f}s".format(data_time.avg)

                    logging.info(log_str)

                    if global_rank == 0 and cfg.wandb.use and step % cfg.train.log_iter == 0:
                        for k, v in log_metrics.items():
                            wandb.log({"train/" + k: v.val}, commit=False)
                        wandb.log({"train/lr": lr_scheduler.get_last_lr()[0]}, commit=False)
                        wandb.log({"train/batch_time": batch_time.val}, commit=False)
                        wandb.log({"train/data_time": data_time.val}, commit=False)
                        wandb.log({"train/cycle": i_cycle}, commit=False)
                        wandb.log({"train/epoch": epoch}, commit=False)
                        wandb.log({"train/iter": step}, commit=False)
                        
                        # Log number of real/syn samples
                        wandb.log({"train/n_real_images": n_real_images}, commit=False)
                        wandb.log({"train/n_real_instances": n_real_instances}, commit=False)
                        wandb.log({"train/n_syn_samples": n_syn_samples}, commit=False)
                        wandb.log({"train/n_syn_instances": n_syn_instances}, commit=False)

                        wandb.log({"train/global_step": global_step}, commit=True)

                    end_time = time.time()
                
                except KeyboardInterrupt as e:
                    logging.info("Keyboard: Saving batch...")
                    # import pickle
                    # pickle.dump(batch, open(f"batch_{global_rank}.pkl", "wb"))
                    raise e
                except RuntimeError as e:
                    logging.info("Runtime: Saving batch...")
                    # import pickle
                    # pickle.dump(batch, open(f"batch_{global_rank}.pkl", "wb"))
                    raise e
            
            # Evaluate here, both syn and real
            # if global_rank == 0 or global_rank == 1:
            model.eval()
            val_model = model.module if is_dist else model
            evaluation_ss(global_rank, val_model, is_dist, val_syn_error_dict, val_real_error_dict, val_syn_loader, val_real_loader, cfg, device, epoch, i_cycle, real_ratios[epoch], global_step)
                # logging.info("Start validation...")
                # model.eval()
                # val_model = model.module if is_dist else model
                # _ = [v.reset() for v in val_syn_error_dict.values()]
                # _ = [v.reset() for v in val_real_error_dict.values()]
                # logging.info("Evaluating the synthetic dataset...")
                # _ = val(val_model, val_syn_loader, device, cfg.test.log_iter, val_syn_error_dict, do_postprocessing=False, use_trimap=False, callback=None, use_temp=False)
            
                # logging.info("Evaluating the real dataset...")
                # _ = val(val_model, val_real_loader, device, cfg.test.log_iter, val_real_error_dict, do_postprocessing=False, use_trimap=False, callback=None, use_temp=False)

                # log_str = "Validation syn data:"
                # for k, v in val_syn_error_dict.items():
                #     log_str += "{}: {:.4f}, ".format(k, v.average())
                # logging.info(log_str)

                # log_str = "Validation real data:"
                # for k, v in val_real_error_dict.items():
                #     log_str += "{}: {:.4f}, ".format(k, v.average())
                # logging.info(log_str)

                # # Log wandb
                # if cfg.wandb.use:
                #     for k, v in val_syn_error_dict.items():
                #         wandb.log({"val/syn_" + k: v.average()}, commit=False)
                #     for k, v in val_real_error_dict.items():
                #         wandb.log({"val/real_" + k: v.average()}, commit=False)
                #     wandb.log({"val/epoch": epoch, 
                #                "val/cycle": i_cycle,
                #                "val/real_ratio": real_ratios[epoch],
                #                "val/global_step": global_step
                #                }, commit=True)
            if global_rank == 0:
                # Save the model
                logging.info("Saving the model...")
                save_dict = {
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'n_real_samples': n_real_images,
                    'n_real_instances': n_real_instances,
                    'n_syn_samples': n_syn_samples,
                    'n_syn_instances': n_syn_instances,
                    'epoch': epoch,
                    'cycle': i_cycle
                }
                save_path = os.path.join(cfg.output_dir, 'last_opt.pth')
                torch.save(save_dict, save_path)

                save_path = os.path.join(cfg.output_dir, 'model_{}_{}.pth'.format(i_cycle, epoch))
                torch.save(val_model.state_dict(), save_path)

        start_epoch = 0
    # while iter < cfg.train.max_iter:
        
    #     for _, batch in enumerate(train_loader):
            
    #         if is_dist:
    #             train_sampler.set_epoch(epoch)

    #         data_time.update(time.time() - end_time)

    #         iter += 1
    #         if iter > cfg.train.max_iter:
    #             break

    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         batch['iter'] = iter
    #         optimizer.zero_grad()
    #         if precision == 16:
    #             with autocast():
    #                 output, loss = model(batch, mem_feat=None)
    #         else:
    #             output, loss = model(batch, mem_feat=None)
    #         if loss is None:
    #             logging.error("Loss is None!")
    #             continue
                
    #         # Store to log_metrics
    #         loss_reduced = loss #reduce_dict(loss)
    #         for k, v in loss_reduced.items():
    #             if k not in log_metrics:
    #                 log_metrics[k] = AverageMeter(k)
    #             log_metrics[k].update(v.item())
            
    #         # Logging
    #         if iter % cfg.train.log_iter == 0:
    #             log_str = "Epoch: {}, Iter: {}/{}".format(epoch, iter, cfg.train.max_iter)
    #             for k, v in log_metrics.items():
    #                 log_str += ", {}: {:.4f}".format(k, v.avg)
    #             log_str += ", lr: {:.6f}".format(lr_scheduler.get_last_lr()[0])
    #             log_str += ", batch_time: {:.4f}s".format(batch_time.avg)
    #             log_str += ", data_time: {:.4f}s".format(data_time.avg)

    #             logging.info(log_str)
    #         if global_rank == 0 and cfg.wandb.use and iter % cfg.train.log_iter == 0:
    #             for k, v in log_metrics.items():
    #                 wandb.log({"train/" + k: v.val}, commit=False)
    #             wandb.log({"train/lr": lr_scheduler.get_last_lr()[0]}, commit=False)
    #             wandb.log({"train/batch_time": batch_time.val}, commit=False)
    #             wandb.log({"train/data_time": data_time.val}, commit=False)
    #             wandb.log({"train/epoch": epoch}, commit=False)
    #             wandb.log({"train/iter": iter}, commit=True)

    #         batch_time.update(time.time() - end_time)

            

    #         if precision == 16:
    #             scaler.scale(loss['total']).backward()
    #         else:
    #             loss['total'].backward()

    #         # Clip norm
    #         if precision == 16:
    #             scaler.unscale_(optimizer)
    #         all_params = itertools.chain(*[x["params"] for x in optimizer.param_groups])
    #         torch.nn.utils.clip_grad_norm_(all_params, 0.01)
            
    #         # Update
    #         logging.debug("Updating")
    #         if precision == 16:
    #             scaler.step(optimizer)
    #             scaler.update()
    #         else:
    #             optimizer.step()

    #         lr_scheduler.step()
            
            
    #         # Visualization
    #         if global_rank == 0 and iter % cfg.train.vis_iter == 0 and cfg.wandb.use:
    #             # Visualize to wandb
    #             try:
    #                 wandb_log_image(batch, output, iter)
    #             except:
    #                 pass
                
    #         # Validation
    #         if iter % cfg.train.val_iter == 0 and (cfg.train.val_dist or (not cfg.train.val_dist and global_rank == 0)):
    #             logging.info("Start validation...")
    #             model.eval()
    #             val_model = model.module if is_dist else model
    #             _ = [v.reset() for v in val_error_dict.values()]
    #             use_temp = cfg.test.temp_aggre and ((not cfg.train.val_dist) or not is_dist)
    #             _ = val(val_model, val_loader, device, cfg.test.log_iter, val_error_dict, do_postprocessing=False, use_trimap=False, callback=None, use_temp=use_temp)

    #             if is_dist and cfg.train.val_dist:
    #                 logging.info("Gathering metrics...")
    #                 # Gather all metrics
    #                 for k, v in val_error_dict.items():
    #                     v.gather_metric(0)
    #             if global_rank == 0:
    #                 log_str = "Validation:"
    #                 for k, v in val_error_dict.items():
    #                     log_str += "{}: {:.4f}, ".format(k, v.average())
    #                 logging.info(log_str)
    #                 # Save best model
    #                 total_error = val_error_dict[cfg.train.val_best_metric].average()
    #                 if total_error < best_score:
    #                     logging.info("Best score changed from {:.4f} to {:.4f}".format(best_score, total_error))
    #                     best_score = total_error
    #                     logging.info("Saving best model...")
    #                     save_path = os.path.join(cfg.output_dir, 'best_model.pth')
    #                     with open(os.path.join(cfg.output_dir,"best_metrics.txt"), 'w') as f:
    #                         f.write("iter: {}\n".format(iter))
    #                         for k, v in val_error_dict.items():
    #                             f.write("{}: {:.4f}\n".format(k, v.average()))
    #                     torch.save(val_model.state_dict(), save_path)
    #                 if cfg.wandb.use:
    #                     for k, v in val_error_dict.items():
    #                         wandb.log({"val/" + k: v.average()}, commit=False)
    #                     wandb.log({"val/epoch": epoch}, commit=False)
    #                     wandb.log({"val/best_error": best_score}, commit=False)
    #                     wandb.log({"val/iter": iter}, commit=True)
    #                 logging.info("Saving last model...")
    #                 save_dict = {
    #                     'optimizer': optimizer.state_dict(),
    #                     'lr_scheduler': lr_scheduler.state_dict(),
    #                     'iter': iter,
    #                     'best_score': best_score
    #                 }
    #                 save_path = os.path.join(cfg.output_dir, 'last_opt.pth')
    #                 torch.save(save_dict, save_path)
    #                 save_path = os.path.join(cfg.output_dir, 'last_model.pth')
    #                 torch.save(val_model.state_dict(), save_path)

    #             model.train()
    #         end_time = time.time()
    #         logging.debug("Loading data")
    #     epoch += 1
