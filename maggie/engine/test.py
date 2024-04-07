
import os
import logging
import copy
import time
from functools import partial

import torch
import numpy as np
import cv2
from torch.utils import data as torch_data

from maggie.dataloader import build_dataset
from maggie.network import build_model
from maggie.utils.dist import AverageMeter
from maggie.utils.postprocessing import reverse_transform_tensor, postprocess
from maggie.utils.metric import build_metric

@torch.no_grad()
def save_visualization(image_names, alpha_names, alphas, transform_info, output, save_dir):
    trans_preds = None
    inc_bin_maps = None
    if 'diff_pred' in output:
        trans_preds = output['diff_pred']
        trans_preds = reverse_transform_tensor(trans_preds, transform_info).cpu().numpy()
    if 'inc_bin_maps' in output:
        inc_bin_maps = output['inc_bin_maps'][0].float()
        inc_bin_maps = reverse_transform_tensor(inc_bin_maps, transform_info).cpu().numpy() > 0.5
        inc_bin_maps = inc_bin_maps.astype('uint8')

    for idx in range(len(image_names)):
        image_name = image_names[idx][0]
        video_name, image_name = image_name.split('/')[-2:]

        # Save alpha pred
        alpha_pred_path = os.path.join(save_dir, video_name)
        os.makedirs(alpha_pred_path, exist_ok=True)
        alpha_pred = (alphas[0, idx] * 255).astype('uint8')
        for inst_id in range(alpha_pred.shape[0]):
            target_path = os.path.join(alpha_pred_path, image_name[:-4])
            if alpha_names is not None:
                target_path = os.path.join(target_path, alpha_names[inst_id][0])
            else:
                if alpha_pred.shape[0] > 1:
                    target_path = os.path.join(target_path, "{:2d}.png".format(inst_id).replace(' ', '0'))
                else:
                    target_path = target_path + ".png"
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            cv2.imwrite(target_path, alpha_pred[inst_id])
                

        if trans_preds is not None:
            # Save trans pred
            trans_pred_path = os.path.join(save_dir, 'diff_pred', video_name)
            os.makedirs(trans_pred_path, exist_ok=True)
            # import pdb; pdb.set_trace()
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

def compute_metrics(all_preds, all_trimap, all_gts, val_error_dict, device, prev_preds=None, prev_trimap=None, prev_gts=None):
    current_metrics = {}

    for k, v in val_error_dict.items():
        cur_trimap = all_trimap
        cur_preds = all_preds
        cur_gts = all_gts
        if k in ['dtSSD', 'MESSDdt']:
            if prev_preds is None:
                continue
            else:
                cur_preds = np.concatenate([prev_preds, all_preds], axis=0)
                cur_gts = np.concatenate([prev_gts, all_gts], axis=0)
                cur_trimap = np.concatenate([prev_trimap, all_trimap], axis=0)

        if k.endswith("_fg"):
            cur_trimap = (all_trimap == 2).astype('float32')
        elif k.endswith("_bg"):
            cur_trimap = (all_trimap == 0).astype('float32')
        elif k.endswith("_unk"):
            cur_trimap = (all_trimap == 1).astype('float32')
        else:
            cur_trimap = None
       
        current_metrics[k] = v.update(cur_preds, cur_gts, trimap=cur_trimap, device=device)
    return current_metrics

@torch.no_grad()
def eval_image(model, val_loader, device, log_iter, val_error_dict, do_postprocessing=False, callback=None, **kwargs):
    
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    end_time = time.time()

    model.eval()

    with torch.no_grad():

        for i, batch in enumerate(val_loader):

            data_time.update(time.time() - end_time)

            # Prepare data
            image_names = batch.pop('image_names')
            alpha_names = None
            if 'alpha_names' in batch:
                alpha_names = batch.pop('alpha_names')
            transform_info = batch.pop('transform_info')
            trimap = batch.pop('trimap').numpy()
            alpha_gt = batch.pop('alpha').numpy()
            skip = batch.pop('skip').numpy()[0]
            batch = {k: v.to(device) for k, v in batch.items()}

            # Ignore the input with no mask guidance
            if batch['mask'].sum() == 0:
                continue
            
            end_time = time.time()

            # Forward pass
            output = model(batch, mem_feat=None)
            
            exec_time = time.time() - end_time
            batch_time.update(exec_time)

            # Postprocessing alpha
            alpha = output['refined_masks']
            alpha = reverse_transform_tensor(alpha, transform_info).cpu().numpy()
            
            # Threshold some high-low values
            alpha[alpha <= 1.0/255.0] = 0.0
            alpha[alpha >= 254.0/255.0] = 1.0

            if do_postprocessing:
                alpha = postprocess(alpha)

            # Compute metrics
            current_metrics = compute_metrics(alpha[:, skip:], trimap[:, skip:], alpha_gt[:, skip:], val_error_dict, device)

            # Logging
            if i % log_iter == 0:
                log_str = "Validation: Iter {}/{}: ".format(i, len(val_loader))
                for k, v in current_metrics.items():
                    log_str += "{} - {:.4f}, ".format(k, v)
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                log_str += 'batch_time: {:.4f}, data_time: {:.4f}, memory: {:.4f} MB'.format(batch_time.avg, data_time.avg, memory)
                logging.info(log_str)
            
            # Callback visualization
            if callback:
                callback(image_names, alpha_names, alpha, transform_info, output)

            end_time = time.time()

    return batch_time.avg, data_time.avg

def get_single_video_metrics(callback, all_image_names, all_preds, transform_info, val_error_dict, all_trimap, all_gts, video_name, device):
    current_metrics = {}

    if callback:
        callback(all_image_names, None, all_preds[None], transform_info, {})
    # Compute the metrics
    for k, v in val_error_dict.items():
        current_trimap = None
        if k.endswith("_fg"):
            current_trimap = (all_trimap[None] == 2).astype('float32')
        elif k.endswith("_bg"):
            current_trimap = (all_trimap[None] == 0).astype('float32')
        elif k.endswith("_unk"):
            current_trimap = (all_trimap[None] == 1).astype('float32')
        current_metrics[k] = v.update(all_preds[None], all_gts[None], trimap=current_trimap, device=device)
    
    log_str = f"{video_name}: "
    for k, v in current_metrics.items():
        log_str += "{} - {:.4f}, ".format(k, v)
    logging.info(log_str)



@torch.no_grad()
def eval_video(model, val_loader, device, log_iter, val_error_dict, do_postprocessing=False, use_trimap=True, callback=None, use_temp=False):
    
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    end_time = time.time()

    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        video_name = None

        all_preds = []
        all_gts = []
        all_trimap = []
        all_image_names = []
        mem_feats = None

        for i, batch in enumerate(val_loader):

            data_time.update(time.time() - end_time)

            image_names = batch.pop('image_names')
            if 'alpha_names' in batch:
                _ = batch.pop('alpha_names')
            
            transform_info = batch.pop('transform_info')
            trimap = batch.pop('trimap').numpy()
            alpha_gt = batch.pop('alpha').numpy()
            
            # Reset if new video
            if image_names[0][0].split('/')[-2] != video_name:

                if len(all_preds) > 0:
                    # get_single_video_metrics(callback, all_image_names, all_preds, transform_info, val_error_dict, all_trimap, all_gts, video_name, device)

                    # Eval the last frame
                    if callback is not None:
                        callback(all_image_names, None, all_preds[None], transform_info, {})
                    # current_metrics = compute_metrics(all_preds[-2:], all_trimap[-2:], all_gts[-2:], val_error_dict, device)
                    # log_str = f"{video_name}: "
                    # for k, v in current_metrics.items():
                    #     log_str += "{} - {:.4f}, ".format(k, v)
                    # logging.info(log_str)

                    # For the last chunk, eval the t and t+1, prev_frames: t-1
                    current_metrics = compute_metrics(all_preds[-2:], all_trimap[-2:], all_gts[-2:], prev_preds[-3:-2], prev_trimaps[-3:-2], prev_gts[-3:-2], val_error_dict, device)
                    log_str = f"{video_name}: "
                    for k, v in current_metrics.items():
                        log_str += "{} - {:.4f}, ".format(k, v)
                    logging.info(log_str)


                    # Free the saving frames
                    all_preds = []
                    all_gts = []
                    all_trimap = []
                    all_image_names = []
                    mem_feats = None
                    torch.cuda.empty_cache()
                
                video_name = image_names[0][0].split('/')[-2]

            batch = {k: v.to(device) for k, v in batch.items()}

            end_time = time.time()
            if batch['mask'].sum() == 0:
                continue
            output = model(batch, mem_feat=mem_feats, mem_query=None, mem_details=None)

            batch_time.update(time.time() - end_time)
                
            alpha = output['refined_masks']
            alpha = reverse_transform_tensor(alpha, transform_info).cpu().numpy()

            # Threshold some high-low values
            alpha[alpha <= 1.0/255.0] = 0.0
            alpha[alpha >= 254.0/255.0] = 1.0

            if do_postprocessing:
                alpha = postprocess(alpha)

            # Fuse results
            # Store all results (3 frames) for the first batch
            if len(all_preds) == 0:
                all_preds = alpha[0]
                all_gts = alpha_gt[0]
                all_trimap = trimap[0]
                all_image_names = image_names
            else:
                # Store the last frame to cumulative GTs and Trimap
                all_gts = np.concatenate([all_gts, alpha_gt[0, 2:]], axis=0)
                all_trimap = np.concatenate([all_trimap, trimap[0, 2:]], axis=0)
                all_image_names += image_names[2:]
                
                # Update the previous frame predictions (t, t+1 in previous batch with t-1, t in the current batch)
                if 'diff_pred_forward' in output:
                    # In the current batch

                    # t-1 (output from previous batch)
                    prev_pred = all_preds[-2]
                    # t + 1 (new output)
                    next_pred = alpha[0, -1]
                    

                    # Get the difference prediction between t-1 -> t and t+1 -> t
                    diff_forward = output['diff_pred_forward']
                    diff_backward = output['diff_pred_backward']
                    diff_forward = reverse_transform_tensor(diff_forward, transform_info).cpu().numpy()
                    diff_backward = reverse_transform_tensor(diff_backward, transform_info).cpu().numpy()
                
                    # Thresholding
                    diff_forward = (diff_forward > 0.5).astype('float32')
                    diff_backward = (diff_backward > 0.5).astype('float32')

                    # Forward propagate from t-1 to t
                    pred_forward01 = prev_pred * (1 - diff_forward[0, 1]) + alpha[0, 1] * diff_forward[0, 1]

                    # Backward propagate from t+1 to t
                    pred_backward21 = next_pred * (1 - diff_backward[0, 1]) + alpha[0, 1] * diff_backward[0, 1]

                    # Check the diff --> update the diff forward --> fused pred based on diff forward
                    diff = np.abs(pred_forward01 - pred_backward21)

                    # Use pred t from the model with pred forward != pred backward
                    pred_forward01[diff > 0.0] = alpha[0, 1][diff > 0.0]

                    # Final result at t
                    fused_pred01 = pred_forward01
                    all_preds[-1] = fused_pred01

                    # For last frame
                    pred_forward12 = fused_pred01 * (1 - diff_forward[0, 2]) + alpha[0, 2] * diff_forward[0, 2]

                    # if use hard fusion
                    
                    all_preds = np.concatenate([all_preds, pred_forward12[None]], axis=0)

                else:
                    # Add the t+1 frame to all_preds
                    all_preds = np.concatenate([all_preds, alpha[0, 2:]], axis=0)

            # Add first frame features to mem_feat
            if mem_feats is None and 'mem_feat' in output:
                if isinstance(output['mem_feat'], tuple):
                    mem_feats = tuple(x[:, 0] for x in output['mem_feat'])
            
            # Save the first frame, delete the previous pred
            # import pdb; pdb.set_trace()
            if callback is not None:
                callback(all_image_names[0:1], None, all_preds[None, 0:1], transform_info, {})

            # Compute the evaluation metrics
            prev_preds = all_preds[-4:-3] if len(all_preds) > 3 else None
            prev_trimaps = all_trimap[-4:-3] if len(all_preds) > 3 else None
            prev_gts = all_gts[-4:-3] if len(all_preds) > 3 else None
            
            current_metrics = compute_metrics(all_preds[-3:-2], all_trimap[-3:-2], all_gts[-3:-2], prev_preds, prev_trimaps, prev_gts, val_error_dict, device)
            log_str = f"{video_name}: "
            for k, v in current_metrics.items():
                log_str += "{} - {:.4f}, ".format(k, v)
            logging.info(log_str)

            # For each chunk, eval the t-1 and dtssd t-1 and t-2
                
            if len(all_preds) > 3:
                all_preds = all_preds[-3:]
                all_gts = all_gts[-3:]
                all_trimap = all_trimap[-3:]
                all_image_names = all_image_names[-3:]
            
            # Logging
            if i % log_iter == 0:
                log_str = "Validation: Iter {}/{}: ".format(i, len(val_loader))
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                log_str += 'batch_time: {:.4f}, data_time: {:.4f}, memory: {:.4f} MB'.format(batch_time.avg, data_time.avg, memory)
                logging.info(log_str)

            end_time = time.time()
    return batch_time.avg, data_time.avg

@torch.no_grad()
def test(cfg, rank=0, is_dist=False):

    # Create dataset
    logging.info("Creating testing dataset...")
    val_dataset = build_dataset(cfg.dataset.test, is_train=False)
    val_sampler = torch_data.DistributedSampler(val_dataset, shuffle=False) if is_dist else None
    
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
    
    # Adding some MAD metrics for each fg, bg, unk
    val_error_dict["MAD_fg"] = copy.deepcopy(val_error_dict['MAD'])
    val_error_dict["MAD_bg"] = copy.deepcopy(val_error_dict['MAD'])
    val_error_dict["MAD_unk"] = copy.deepcopy(val_error_dict['MAD'])

    # Start testing
    logging.info("Start testing...")
    val_fn = eval_video if cfg.dataset.test.name == 'VIM' else eval_image
    batch_time, data_time = val_fn(model, val_loader, device, cfg.test.log_iter, \
                                val_error_dict, do_postprocessing=cfg.test.postprocessing, \
                                    callback=partial(save_visualization, save_dir=cfg.test.save_dir) if cfg.test.save_results else None, 
                                    use_trimap=cfg.test.use_trimap, use_temp=cfg.test.temp_aggre)
    
    logging.info("Testing done!")

    if is_dist:
        logging.info("Gathering metrics...")
        # Gather all metrics
        for k, v in val_error_dict.items():
            v.gather_metric(0)
    
    if rank == 0:
        logging.info("Metrics:")
        metric_str = ""
        plain_str = ""
        for k, v in val_error_dict.items():
            metric_str += "{}: {}\n".format(k, v.average())
            plain_str += str(v.average()) + ","
        logging.info(metric_str)
        logging.info(plain_str)
        logging.info('batch_time: {:.4f}, data_time: {:.4f}'.format(batch_time, data_time))


