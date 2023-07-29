
import os
import random
import time
import torch
import logging
import numpy as np
import cv2
from torch.utils import data as torch_data
from vm2m.dataloader import build_dataset
from vm2m.network import build_model
from vm2m.utils.dist import AverageMeter
from vm2m.utils.postprocessing import reverse_transform_tensor, postprocess
from vm2m.utils.metric import build_metric

@torch.no_grad()
def save_visualization(save_dir, image_names, alphas, transform_info, output):
    trans_preds = None
    inc_bin_maps = None
    if 'trans_preds' in output:
        trans_preds = output['trans_preds'][0]
        trans_preds = reverse_transform_tensor(trans_preds, transform_info).sigmoid().cpu().numpy()
    if 'inc_bin_maps' in output:
        inc_bin_maps = output['inc_bin_maps'][0].float()
        inc_bin_maps = reverse_transform_tensor(inc_bin_maps, transform_info).cpu().numpy() > 0.5
        inc_bin_maps = inc_bin_maps.astype('uint8')

    for idx in range(len(image_names)):
        image_name = image_names[idx][0]
        video_name, image_name = image_name.split('/')[-2:]

        # Save alpha pred
        alpha_pred_path = os.path.join(save_dir, 'alpha_pred', video_name)
        os.makedirs(alpha_pred_path, exist_ok=True)
        alpha_pred = (alphas[0, idx] * 255).astype('uint8')
        for inst_id in range(alpha_pred.shape[0]):
            postfix = '_inst%d' % inst_id if alpha_pred.shape[0] > 1 else ''
            if not os.path.isfile(os.path.join(alpha_pred_path, image_name[:-4] + postfix + image_name[-4:])):    
                cv2.imwrite(os.path.join(alpha_pred_path, image_name[:-4] + postfix + image_name[-4:]), alpha_pred[inst_id])

        if trans_preds is not None:
            # Save trans pred
            trans_pred_path = os.path.join(save_dir, 'trans_pred', video_name)
            os.makedirs(trans_pred_path, exist_ok=True)

            trans_pred = (trans_preds[0, idx, 0] * 255).astype('uint8')
            if not os.path.isfile(os.path.join(trans_pred_path, image_name)):
                cv2.imwrite(os.path.join(trans_pred_path, image_name), trans_pred)

        # Save inc binary pred
        if inc_bin_maps is not None:
            inc_bin_path = os.path.join(save_dir, 'inc_pred', video_name)
            os.makedirs(inc_bin_path, exist_ok=True)
            inc_bin_pred = (inc_bin_maps[0, idx, 0] * 255).astype('uint8')
            if not os.path.isfile(os.path.join(inc_bin_path, image_name)):
                cv2.imwrite(os.path.join(inc_bin_path, image_name), inc_bin_pred)

@torch.no_grad()
def val(model, val_loader, device, log_iter, val_error_dict, do_postprocessing=False, use_trimap=True, callback=None):
    
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    end_time = time.time()

    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):

            data_time.update(time.time() - end_time)
            
            # if i < 681:
            #     continue
            # import pdb; pdb.set_trace()

            image_names = batch.pop('image_names')
            transform_info = batch.pop('transform_info')
            trimap = batch.pop('trimap').numpy()
            alpha_gt = batch.pop('alpha').numpy()
            skip = batch.pop('skip').numpy()[0]

            batch = {k: v.to(device) for k, v in batch.items()}

            end_time = time.time()
            if batch['mask'].sum() == 0:
                continue
            output = model(batch)
            batch_time.update(time.time() - end_time)

            
            alpha = output['refined_masks']
            alpha = alpha #.cpu().numpy()
            alpha = reverse_transform_tensor(alpha, transform_info).cpu().numpy()
            if do_postprocessing:
                alpha = postprocess(alpha)

            current_metrics = {}
            for k, v in val_error_dict.items():
                logging.debug(f"updating {k}...")
                current_trimap = trimap[:, skip:] if use_trimap else None
                current_metrics[k] = v.update(alpha[:, skip:], alpha_gt[:, skip:], trimap=current_trimap)
                logging.debug(f"Done {k}!")

            # Logging
            if i % log_iter == 0:
                log_str = "Validation: Iter {}/{}: ".format(i, len(val_loader))
                for k, v in current_metrics.items():
                    log_str += "{} - {:.4f}, ".format(k, v)
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                log_str += 'batch_time: {:.4f}, data_time: {:.4f}, memory: {:.4f} MB'.format(batch_time.avg, data_time.avg, memory)
                logging.info(log_str)
            
            # Visualization
            if callback:
                callback(image_names, alpha, transform_info, output)

            end_time = time.time()
    return batch_time.avg, data_time.avg

@torch.no_grad()
def test(cfg, rank=0, is_dist=False):

    # Create dataset
    logging.info("Creating testing dataset...")
    val_dataset = build_dataset(cfg.dataset.test, is_train=False)
    val_sampler = torch_data.DistributedSampler(val_dataset, shuffle=False) if is_dist else None

    # val_dataset.frame_ids = val_dataset.frame_ids[681:]
    
    val_loader = torch_data.DataLoader(
        val_dataset, batch_size=cfg.test.batch_size, shuffle=False, pin_memory=False,
        sampler=val_sampler,
        num_workers=cfg.test.num_workers)
    
    device = f"cuda:{rank}"

    # Build model
    logging.info("Building model...")
    model = build_model(cfg.model)
    model = model.to(device)

    # Load pretrained model
    assert os.path.isfile(cfg.model.weights), "Cannot find pretrained model at {}".format(cfg.model.weights)
    
    logging.info("Loading pretrained model from {}".format(cfg.model.weights))
    state_dict = torch.load(cfg.model.weights, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        logging.warn("Missing keys: {}".format(missing_keys))
        logging.warn("Unexpected keys: {}".format(unexpected_keys))
    
    num_parameters = sum([p.numel() for p in model.parameters()])

    logging.info("Number of parameters: {}".format(num_parameters))
    
    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Build metric
    val_error_dict = build_metric(cfg.test.metrics)

    # Start testing
    logging.info("Start testing...")
    if cfg.test.save_results:
        def callback_vis(image_names, alpha, transform_info, output):
            save_visualization(cfg.test.save_dir, image_names, alpha, transform_info, output)
    else:
        callback_vis = None

    batch_time, data_time = val(model, val_loader, device, cfg.test.log_iter, \
                                val_error_dict, do_postprocessing=cfg.test.postprocessing, \
                                    callback=callback_vis, use_trimap=cfg.test.use_trimap)
    
    logging.info("Testing done!")

    if is_dist:
        logging.info("Gathering metrics...")
        # Gather all metrics
        for k, v in val_error_dict.items():
            v.gather_metric(0)
    
    if rank == 0:
        logging.info("Metrics:")
        metric_str = ""
        for k, v in val_error_dict.items():
            metric_str += "{}: {}\n".format(k, v.average())
        logging.info(metric_str)
        logging.info('batch_time: {:.4f}, data_time: {:.4f}'.format(batch_time, data_time))


