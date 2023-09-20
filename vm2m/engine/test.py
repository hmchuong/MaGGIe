
from functools import partial
import os
import copy
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
def save_visualization(image_names, alpha_names, alphas, transform_info, output, save_dir):
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
        # import pdb; pdb.set_trace()
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
            # postfix = '_inst%d' % inst_id if alpha_pred.shape[0] > 1 else ''
            # if not os.path.isfile(os.path.join(alpha_pred_path, image_name[:-4] + postfix + image_name[-4:])):    
            #     cv2.imwrite(os.path.join(alpha_pred_path, image_name[:-4] + postfix + image_name[-4:]), alpha_pred[inst_id])
                

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
def val(model, val_loader, device, log_iter, val_error_dict, do_postprocessing=False, use_trimap=True, callback=None, use_temp=False):
    
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    end_time = time.time()

    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        mem_feat = []
        mem_query = None
        mem_details = None
        memory_interval = 5
        n_mem = 1
        video_name = None

        prev_pred = None
        prev_gt = None
        prev_trimap = None

        for i, batch in enumerate(val_loader):

            data_time.update(time.time() - end_time)
            
            # if i < 117:
            #     continue
            # import pdb; pdb.set_trace()

            image_names = batch.pop('image_names')
            alpha_names = None
            if 'alpha_names' in batch:
                alpha_names = batch.pop('alpha_names')
            
            transform_info = batch.pop('transform_info')
            trimap = batch.pop('trimap').numpy()
            alpha_gt = batch.pop('alpha').numpy()
            skip = batch.pop('skip').numpy()[0]
            
            # Reset if new video
            if image_names[0][0].split('/')[-2] != video_name:

                video_name = image_names[0][0].split('/')[-2]
                mem_feat = []
                mem_query = None
                mem_details = None
                processed_frames = 0
                prev_gt = None
                prev_pred = None
                prev_trimap = None

            batch = {k: v.to(device) for k, v in batch.items()}

            end_time = time.time()
            if batch['mask'].sum() == 0:
                continue
            # Adding prev_feat
            # prev_feat = {}
            prev_mem = []
            if len(mem_feat) >= 1 and use_temp:
                prev_mem.append(mem_feat[-1])
                m_i = 2
                while len(mem_feat) - m_i >= 0:
                    if m_i % memory_interval == 0:
                        prev_mem.append(mem_feat[-m_i])
                    m_i+= 1
                
            output = model(batch, mem_feat=prev_mem, mem_query=mem_query, mem_details=mem_details)

            batch_time.update(time.time() - end_time)
            processed_frames += 1

            # Save memory frames
            if use_temp and 'mem_feat' in output:
                mem_feat.append(output['mem_feat'].unsqueeze(1))
                if len(mem_feat) > memory_interval * n_mem:
                    mem_feat = mem_feat[-(memory_interval * n_mem):]
                mem_query = output['mem_queries']
                mem_details = output['mem_details']
            # if 'embedding' in output:
            #     memory_frames.append(output['embedding'])
            #     if len(memory_frames) > memory_interval:
            #         memory_frames = memory_frames[-memory_interval:]
            #         # print("Cut down memory frames")
            # if 'prev_mask' in output:
            #     memory_prev_mask.append(output['prev_mask'])
            #     if len(memory_prev_mask) > memory_interval:
            #         memory_prev_mask = memory_prev_mask[-memory_interval:]
                
            
            alpha = output['refined_masks']
            alpha = alpha #.cpu().numpy()
            alpha = reverse_transform_tensor(alpha, transform_info).cpu().numpy()
            if do_postprocessing:
                alpha = postprocess(alpha)

            # DEBUG: Load masks for instmatt
            # import glob
            # all_alpha_paths = sorted(glob.glob(image_names[0][0].replace('/images/', '/instmatt/').replace(".jpg", "/*.png")))
            # all_alphas = []
            # for alpha_path in all_alpha_paths:
            #     all_alphas.append(cv2.imread(alpha_path, 0))
            # alpha = np.stack(all_alphas, axis=0)[None, None] / 255.0

            current_metrics = {}
            for k, v in val_error_dict.items():
                if k in ['dtSSD', 'MESSDdt', 'dtSSD_trimap', 'MESSDdt_trimap'] and use_temp:
                    if prev_gt is None:
                        continue
                    current_trimap = np.stack([prev_trimap, trimap[:, -1]], axis=1) if k.endswith("_trimap") else None
                    current_pred = np.stack([prev_pred, alpha[:, -1]], axis=1)
                    current_gt = np.stack([prev_gt, alpha_gt[:, -1]], axis=1)
                    current_metrics[k] = v.update(current_pred, current_gt, trimap=current_trimap)
                    continue
                logging.debug(f"updating {k}...")
                current_trimap = None
                if k.endswith("_fg"):
                    current_trimap = (trimap[:, skip:] == 2).astype('float32')
                elif k.endswith("_bg"):
                    current_trimap = (trimap[:, skip:] == 0).astype('float32')
                elif k.endswith("_unk"):
                    current_trimap = (trimap[:, skip:] == 1).astype('float32')
                # current_trimap = trimap[:, skip:] if k.endswith("_trimap") else None
                current_metrics[k] = v.update(alpha[:, skip:], alpha_gt[:, skip:], trimap=current_trimap)
                logging.debug(f"Done {k}!")
            
            # all_preds.append(alpha[:, skip:])
            # all_gts.append(alpha_gt[:, skip:])
            # all_trimaps.append(trimap[:, skip:])
            prev_gt = alpha_gt[:, -1]
            prev_pred = alpha[:, -1]
            prev_trimap = trimap[:, -1]

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
                callback(image_names, alpha_names, alpha, transform_info, output)

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
    val_error_dict["MAD_fg"] = copy.deepcopy(val_error_dict['MAD'])
    val_error_dict["MAD_bg"] = copy.deepcopy(val_error_dict['MAD'])
    val_error_dict["MAD_unk"] = copy.deepcopy(val_error_dict['MAD'])

    # Start testing
    logging.info("Start testing...")
    # if cfg.test.save_results:
    #     def callback_vis(image_names, alpha, transform_info, output):
    #         save_visualization(cfg.test.save_dir, image_names, alpha, transform_info, output)
    # else:
    #     callback_vis = None

    batch_time, data_time = val(model, val_loader, device, cfg.test.log_iter, \
                                val_error_dict, do_postprocessing=cfg.test.postprocessing, \
                                    callback=partial(save_visualization, save_dir=cfg.test.save_dir) if cfg.test.save_results else None, use_trimap=cfg.test.use_trimap, use_temp=cfg.test.temp_aggre)
    
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


