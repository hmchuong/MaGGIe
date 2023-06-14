
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
from .optim import build_optim_lr_scheduler

def val(model, val_loader, device, log_iter=30):
    model.eval()
    val_error_dict = {
        'sad': AverageMeter(),
        'mse': AverageMeter()
    }

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            del batch['image_names']
            del batch['transform_info']
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            alpha = outputs['refined_masks']
            b, n_f, n_i = alpha.shape[:3]
            sad = torch.abs(alpha - batch['alpha']).sum() / (b * n_f * n_i) 
            mse = torch.pow(alpha - batch['alpha'], 2).sum() / (b * n_f * n_i)
            val_error_dict['sad'].update(sad.item() / 1000)
            val_error_dict['mse'].update(mse.item() / 1000)

            if i % log_iter == 0:
                logging.info("Validation: Iter {}/{}: SAD: {:.4f}, MSE: {:.4f}".format(
                    i, len(val_loader), val_error_dict['sad'].avg, val_error_dict['mse'].avg))
    return val_error_dict

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
    if rank == 0:
        logging.info("Creating val dataset...")
        val_dataset = build_dataset(cfg.dataset.test, is_train=False)
        val_loader = torch_data.DataLoader(
            val_dataset, batch_size=cfg.test.batch_size, shuffle=False, pin_memory=True,
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
        model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[rank], find_unused_parameters=True)

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
        logging.warn("Missing keys: {}".format(missing_keys))
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

    # Start training
    logging.info("Start training...")
    model.train()
    epoch = len(train_loader) // iter
    while iter < cfg.train.max_iter:
        
        for _, batch in enumerate(train_loader):
            if is_dist:
                train_sampler.set_epoch(epoch)

            data_time.update(time.time() - end_time)

            iter += 1
            if iter > cfg.train.max_iter:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            output, loss = model(batch)
            if loss is None:
                continue
            loss_reduced = reduce_dict(loss)

            loss['total'].backward()

            # Store to log_metrics
            for k, v in loss_reduced.items():
                if k not in log_metrics:
                    log_metrics[k] = AverageMeter(k)
                log_metrics[k].update(v.item())

            optimizer.step()
            lr_scheduler.step()

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
                # TODO: Visualize to wandb
                
                # Log transition_preds
                log_images = []
                image = batch['image'][0,0].cpu()
                image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                image = (image * 255).permute(1, 2, 0).numpy().astype(np.uint8)
                log_images.append(wandb.Image(image, caption="image"))

                alpha_gt = (batch['alpha'][0,0,0] * 255).cpu().numpy().astype('uint8')
                log_images.append(wandb.Image(alpha_gt, caption="alpha_gt"))

                alpha_pred = (output['refined_masks'][0,0,0] * 255).detach().cpu().numpy().astype('uint8')
                log_images.append(wandb.Image(alpha_pred, caption="alpha_pred"))

                mask_gt = (batch['mask'][0,0,0] * 255).detach().cpu().numpy().astype('uint8')
                log_images.append(wandb.Image(mask_gt, caption="mask_gt"))
                
                for i, trans_pred in enumerate(output['trans_preds']):
                    trans_pred = (trans_pred[0,0,0].sigmoid() * 255).detach().cpu().numpy().astype('uint8')
                    log_images.append(wandb.Image(trans_pred, caption='transition_pred_' + str(i)))
                
                trans_gt = (batch['transition'][0,0,0] * 255).cpu().numpy().astype('uint8')
                log_images.append(wandb.Image(trans_gt, caption="transition_gt"))
                
                for i, inc_bin_map in enumerate(output['inc_bin_maps']):
                    inc_bin_map = (inc_bin_map[0,0,0] * 255).detach().cpu().numpy().astype('uint8')
                    log_images.append(wandb.Image(inc_bin_map, caption='inc_bin_map_' + str(i)))
                wandb.log({"examples/all": log_images}, step=iter, commit=True)
            
            # Validation
            if iter % cfg.train.val_iter == 0 and rank == 0:
                logging.info("Start validation...")
                model.eval()
                val_model = model.module if is_dist else model
                val_metrics = val(val_model, val_loader, device)
                
                logging.info("Validation: SAD: {:.4f}, MSE: {:.4f}".format(val_metrics['sad'].avg, val_metrics['mse'].avg))
                if cfg.wandb.use:
                    for k, v in val_metrics.items():
                        wandb.log({"val/" + k: v.avg}, commit=False)
                    wandb.log({"val/epoch": epoch}, commit=False)
                    wandb.log({"val/iter": iter}, commit=True)
                
                # Save best model
                total_error = val_metrics['sad'].avg + val_metrics['mse'].avg
                if total_error < best_score:
                    logging.info("Best score changed from {:.4f} to {:.4f}".format(best_score, total_error))
                    best_score = total_error
                    logging.info("Saving best model...")
                    save_path = os.path.join(cfg.output_dir, 'best_model.pth')
                    with open(os.path.join(cfg.output_dir,"best_metrics.txt"), 'w') as f:
                        f.write("iter: {}\n".format(iter))
                        f.write("sad: {:.4f}\n".format(val_metrics['sad'].avg))
                        f.write("mse: {:.4f}\n".format(val_metrics['mse'].avg))
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
        
        epoch += 1




    # TODO: Implement training
    # TODO: Implement validation
    # import pdb; pdb.set_trace()


