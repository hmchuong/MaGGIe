
import os
import random
import time
import torch
import logging
import numpy as np
import wandb
from torch.utils import data as torch_data
from vm2m.dataloader import build_dataset
from vm2m.network import build_model
from vm2m.utils.dist import AverageMeter, reduce_dict
from vm2m.utils.metric import build_metric

from .optim import build_optim_lr_scheduler
from .test import val

def log_alpha(tensor, tag, index=0):
    alpha = tensor[0,index,0].detach().cpu().numpy()
    alpha = (alpha * 255).astype('uint8')
    return wandb.Image(alpha, caption=tag)

def wandb_log_image(batch, output, iter):
    # Log transition_preds
    log_images = []
    index = random.randint(0, batch['image'].shape[1] - 1)
    image = batch['image'][0,index].cpu()
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = (image * 255).permute(1, 2, 0).numpy().astype(np.uint8)
    log_images.append(wandb.Image(image, caption="image"))

    log_images.append(log_alpha(batch['alpha'], 'alpha_gt', index))
    # alpha_gt = (batch['alpha'][0,0,0] * 255).cpu().numpy().astype('uint8')
    # log_images.append(wandb.Image(alpha_gt, caption="alpha_gt"))

    log_images.append(log_alpha(output['refined_masks'], 'alpha_pred', index))
    # alpha_pred = (output['refined_masks'][0,0,0] * 255).detach().cpu().numpy().astype('uint8')
    # log_images.append(wandb.Image(alpha_pred, caption="alpha_pred"))

    log_images.append(log_alpha(batch['mask'], 'mask_gt', index))
    # mask_gt = (batch['mask'][0,0,0] * 255).detach().cpu().numpy().astype('uint8')
    # log_images.append(wandb.Image(mask_gt, caption="mask_gt"))
    
    # For VM2M
    if 'trans_preds' in output:
        for i, trans_pred in enumerate(output['trans_preds']):
            # trans_pred = (trans_pred[0,0,0].sigmoid() * 255).detach().cpu().numpy().astype('uint8')
            # log_images.append(wandb.Image(trans_pred, caption='transition_pred_' + str(i)))
            log_images.append(log_alpha(trans_pred.sigmoid(), 'transition_pred_' + str(i), index))
        log_images.append(log_alpha(batch['transition'], 'transition_gt', index))
        # trans_gt = (batch['transition'][0,0,0] * 255).cpu().numpy().astype('uint8')
        # log_images.append(wandb.Image(trans_gt, caption="transition_gt"))

    if 'inc_bin_maps' in output:
        for i, inc_bin_map in enumerate(output['inc_bin_maps']):
            # inc_bin_map = (inc_bin_map[0,0,0] * 255).detach().cpu().numpy().astype('uint8')
            # log_images.append(wandb.Image(inc_bin_map, caption='inc_bin_map_' + str(i)))
            log_images.append(log_alpha(inc_bin_map, 'inc_bin_map_gt_' + str(i), index))
    # For MGM: logging some intermediate results
    if 'alpha_os1' in output:
        log_images.append(log_alpha(output['alpha_os1'], 'alpha_os1_pred', index))
        # alpha_pred = (output['alpha_os1'][0,0,0] * 255).detach().cpu().numpy().astype('uint8')
        # log_images.append(wandb.Image(alpha_pred, caption="alpha_os1_pred"))
    if 'alpha_os4' in output:
        log_images.append(log_alpha(output['alpha_os4'], 'alpha_os4_pred', index))
    if 'alpha_os8' in output:
        log_images.append(log_alpha(output['alpha_os8'], 'alpha_os8_pred', index))
    wandb.log({"examples/all": log_images}, step=iter, commit=True)

def train(cfg, rank, is_dist=False):
    
    device = f'cuda:{rank}'

    # Create dataset
    logging.info("Creating train dataset...")
    train_dataset = build_dataset(cfg.dataset.train, is_train=True, random_seed=cfg.train.seed)

    # Create dataloader
    if is_dist:
        train_sampler = torch_data.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    g = torch.Generator()
    g.manual_seed(cfg.train.seed)

    train_loader = torch_data.DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.train.num_workers,
        pin_memory=True, sampler=train_sampler,
        generator=g)
    
    # Validate only at rank 0
    # if rank == 0:
    logging.info("Creating val dataset...")
    val_dataset = build_dataset(cfg.dataset.test, is_train=False)
    val_sampler = torch_data.DistributedSampler(val_dataset, shuffle=False) if is_dist else None
    val_loader = torch_data.DataLoader(
        val_dataset, batch_size=cfg.test.batch_size, shuffle=False, pin_memory=True,
        sampler=val_sampler,
        num_workers=cfg.test.num_workers)
    
    # Build model
    logging.info("Building model...")
    model = build_model(cfg.model)
    model = model.to(device)
    training_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logging.info("Number of trainable parameters: {}".format(training_params))

    # Define optimizer and lr scheduler
    logging.info("Building optimizer and lr scheduler...")
    optimizer, lr_scheduler = build_optim_lr_scheduler(cfg, model)

    if is_dist:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        having_unused_params = False
        if cfg.model.arch in ['VM2M', 'VM2M0711']:
            having_unused_params = True
        model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[rank], find_unused_parameters=having_unused_params)

    epoch = 0
    iter = 0
    best_score = 99999999999 # SAD?

    # Load pretrained model
    if os.path.isfile(cfg.model.weights):
        logging.info("Loading pretrained model from {}".format(cfg.model.weights))
        state_dict = torch.load(cfg.model.weights, map_location=device)
        if is_dist:
            missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0:
            logging.warn("Missing keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            logging.warn("Unexpected keys: {}".format(unexpected_keys))

    # Resume model from last checkpoint
    if cfg.train.resume != '':
        if os.path.isdir(cfg.train.resume):
            logging.info("Resuming model from {}".format(cfg.train.resume))
            state_dict = torch.load(os.path.join(cfg.train.resume, 'last_model.pth'), map_location=device)
            opt_dict = torch.load(os.path.join(cfg.train.resume, 'last_opt.pth'), map_location=device)
            if is_dist:
                model.module.load_state_dict(state_dict, strict=True)
            else:
                model.load_state_dict(state_dict, strict=True)
            
            # Load optimizer and lr_scheduler
            optimizer.load_state_dict(opt_dict['optimizer'])
            lr_scheduler.load_state_dict(opt_dict['lr_scheduler'])

            # Load epoch, iteration, best score
            iter = opt_dict['iter']
            best_score = opt_dict['best_score']
            logging.info("Resuming from epoch {}, iter {}, best score {}".format(epoch, iter, best_score))
        else:
            raise ValueError("Cannot resume model from {}".format(cfg.train.resume))
    
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    
    log_metrics = {}
    end_time = time.time()

    # Build validation metrics
    val_error_dict = build_metric(cfg.train.val_metrics)
    assert len(val_error_dict) > 0, "No validation metrics found!"
    assert cfg.train.val_best_metric in val_error_dict, "Best validation metric not found!"

    # Start training
    logging.info("Start training...")
    model.train()
    logging.debug("Iter: {}, len dataloader: {}".format(iter, len(train_loader)))
    epoch =  iter // len(train_loader)
    while iter < cfg.train.max_iter:
        
        for _, batch in enumerate(train_loader):
            logging.debug("Loaded data")
            if is_dist:
                train_sampler.set_epoch(epoch)

            data_time.update(time.time() - end_time)

            iter += 1
            if iter > cfg.train.max_iter:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            batch['iter'] = iter
            optimizer.zero_grad()
            # try:
                # if iter == 85 and rank == 0:
                #     from pudb.remote import set_trace
                #     set_trace()
            output, loss = model(batch)
            if loss is None:
                logging.error("Loss is None!")
                continue
            # except ValueError as e:
            #     logging.error("ValueError: {}".format(e))
            #     continue
            logging.debug("Reducing loss")
            loss_reduced = loss #reduce_dict(loss)

            logging.debug("Backwarding")
            loss['total'].backward()

            logging.debug("Storing log metrics")
            # Store to log_metrics
            for k, v in loss_reduced.items():
                if k not in log_metrics:
                    log_metrics[k] = AverageMeter(k)
                log_metrics[k].update(v.item())

            logging.debug("Optimizing")
            optimizer.step()
            logging.debug("Updating lr scheduler")
            lr_scheduler.step()
            logging.debug("Done batch")
            batch_time.update(time.time() - end_time)

            # Logging
            if iter % cfg.train.log_iter == 0 and rank == 0:
                log_str = "Epoch: {}, Iter: {}/{}".format(epoch, iter, cfg.train.max_iter)
                for k, v in log_metrics.items():
                    log_str += ", {}: {:.4f}".format(k, v.avg)
                log_str += ", lr: {:.6f}".format(lr_scheduler.get_last_lr()[0])
                log_str += ", batch_time: {:.4f}s".format(batch_time.avg)
                log_str += ", data_time: {:.4f}s".format(data_time.avg)

                logging.info(log_str)

                # TODO: log to wandb
                if cfg.wandb.use:
                    for k, v in log_metrics.items():
                        wandb.log({"train/" + k: v.val}, commit=False)
                    wandb.log({"train/lr": lr_scheduler.get_last_lr()[0]}, commit=False)
                    wandb.log({"train/batch_time": batch_time.val}, commit=False)
                    wandb.log({"train/data_time": data_time.val}, commit=False)
                    wandb.log({"train/epoch": epoch}, commit=False)
                    wandb.log({"train/iter": iter}, commit=True)
            
            # Visualization
            if iter % cfg.train.vis_iter == 0 and rank == 0 and cfg.wandb.use:
                # Visualize to wandb
                wandb_log_image(batch, output, iter)  
            
            # Validation
            if iter % cfg.train.val_iter == 0:
                logging.info("Start validation...")
                model.eval()
                val_model = model.module if is_dist else model
                _ = [v.reset() for v in val_error_dict.values()]
                _ = val(val_model, val_loader, device, cfg.train.log_iter, val_error_dict, False, None)

                if is_dist:
                    logging.info("Gathering metrics...")
                    # Gather all metrics
                    for k, v in val_error_dict.items():
                        v.gather_metric(0)
                if rank == 0:
                    log_str = "Validation:"
                    for k, v in val_error_dict.items():
                        log_str += "{}: {:.4f}, ".format(k, v.average())
                    if cfg.wandb.use:
                        for k, v in val_error_dict.items():
                            wandb.log({"val/" + k: v.average()}, commit=False)
                        wandb.log({"val/epoch": epoch}, commit=False)
                        wandb.log({"val/iter": iter}, commit=True)
                    
                    # Save best model
                    total_error = val_error_dict[cfg.train.val_best_metric].average()
                    if total_error < best_score:
                        logging.info("Best score changed from {:.4f} to {:.4f}".format(best_score, total_error))
                        best_score = total_error
                        logging.info("Saving best model...")
                        save_path = os.path.join(cfg.output_dir, 'best_model.pth')
                        with open(os.path.join(cfg.output_dir,"best_metrics.txt"), 'w') as f:
                            f.write("iter: {}\n".format(iter))
                            for k, v in val_error_dict.items():
                                f.write("{}: {:.4f}\n".format(k, v.average()))
                        torch.save(val_model.state_dict(), save_path)
                    
                    logging.info("Saving last model...")
                    save_dict = {
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'iter': iter,
                        'best_score': best_score
                    }
                    save_path = os.path.join(cfg.output_dir, 'last_opt.pth')
                    torch.save(save_dict, save_path)
                    save_path = os.path.join(cfg.output_dir, 'last_model.pth')
                    torch.save(val_model.state_dict(), save_path)

                model.train()
            end_time = time.time()
            logging.debug("Loading data")
        epoch += 1




    # TODO: Implement training
    # TODO: Implement validation
    # import pdb; pdb.set_trace()


