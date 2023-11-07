
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
    if 'diff_pred' in output:
        trans_preds = output['diff_pred']
        trans_preds = reverse_transform_tensor(trans_preds, transform_info).cpu().numpy() #.sigmoid().cpu().numpy()
    if 'inc_bin_maps' in output:
        inc_bin_maps = output['inc_bin_maps'][0].float()
        inc_bin_maps = reverse_transform_tensor(inc_bin_maps, transform_info).cpu().numpy() > 0.5
        inc_bin_maps = inc_bin_maps.astype('uint8')

    for idx in range(len(image_names)):
        image_name = image_names[idx][0]
        video_name, image_name = image_name.split('/')[-2:]

        # Save alpha pred
        # alpha_pred_path = os.path.join(save_dir, 'alpha_pred', video_name)
        alpha_pred_path = os.path.join(save_dir, video_name)
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

@torch.no_grad()
def val_image(model, val_loader, device, log_iter, val_error_dict, do_postprocessing=False, use_trimap=True, callback=None, use_temp=False):
    
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    end_time = time.time()

    model.eval()
    torch.cuda.empty_cache()

    target_files = set([
        "unsplash_middle_saeed-karimi-JrrWC7Qcmhs-unsplash",
        "google_easy_c3668dda4d46436097b6c5d58153e7de",
        "google_middle_95e6da3a2f164a80b68abdd8058aa696",
        "google_middle_62604586fc08499ca4d346900c338bc8",
        "google_middle_44440b07b2d5459ea2053cd3ee9f9406",
        "google_middle_2db0e2572a654f55947d88d411c3cb61",
        "celebrity_middle_d3874137945e41319d93448910b8bcdd",
        "celebrity_easy_e129df9159a64841b62c588ddc5730c9",
        "celebrity_middle_1650c9fd73ae4313bae6a598a7d7072e",
        "celebrity_middle_8f47dc49975f4fc484224fdec092253f",
        "celebrity_easy_c0c60daf04cf41e3bb5752d932944dd9",
        "celebrity_middle_2b9db99c95e54d918f2a0795515ef271",
        "celebrity_easy_b71d8703a11240a2aba6babc1193c2e4",
        "Pexels_middle_pexels-photo-5896435",
        "Pexels_easy_pexels-photo-5618157",
        "celebrity_middle_1d3504849baa441fb6367d178c892423"
    ])
    with torch.no_grad():

        for i, batch in enumerate(val_loader):

            data_time.update(time.time() - end_time)

            image_names = batch.pop('image_names')
            alpha_names = None
            if 'alpha_names' in batch:
                alpha_names = batch.pop('alpha_names')
            
            # if image_names[0][0].split('/')[-1].replace(".jpg", "") in target_files:
            #     continue
            transform_info = batch.pop('transform_info')
            trimap = batch.pop('trimap').numpy()
            alpha_gt = batch.pop('alpha').numpy()
            skip = batch.pop('skip').numpy()[0]

            batch = {k: v.to(device) for k, v in batch.items()}

            end_time = time.time()
            if batch['mask'].sum() == 0:
                continue
            output = model(batch, mem_feat=None)

            batch_time.update(time.time() - end_time)
                
            alpha = output['refined_masks']

            alpha = reverse_transform_tensor(alpha, transform_info).cpu().numpy()
        
            # Threshold some high-low values
            alpha[alpha <= 1.0/255.0] = 0.0
            alpha[alpha >= 254.0/255.0] = 1.0

            if do_postprocessing:
                alpha = postprocess(alpha)

            current_metrics = {}
            for k, v in val_error_dict.items():

                current_trimap = None
                if k.endswith("_fg"):
                    current_trimap = (trimap[:, skip:] == 2).astype('float32')
                elif k.endswith("_bg"):
                    current_trimap = (trimap[:, skip:] == 0).astype('float32')
                elif k.endswith("_unk"):
                    current_trimap = (trimap[:, skip:] == 1).astype('float32')
                
                current_metrics[k] = v.update(alpha[:, skip:], alpha_gt[:, skip:], trimap=current_trimap, device=device)
                # if k == "MAD" and current_metrics[k] > 10.0:
                #     print(image_names[0][0])
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
                callback(image_names, alpha_names, alpha, transform_info, output)

            end_time = time.time()
    return batch_time.avg, data_time.avg

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) 
def gen_roi_mask(prev_pred, curr_pred):
    prev_dilated = cv2.dilate((prev_pred.transpose(1,2,0) > 0.5).astype('uint8') * 255, kernel) > 0
    cur_dilated = cv2.dilate((curr_pred.transpose(1,2,0) > 0.5).astype('uint8')  * 255, kernel) > 0
    union = prev_dilated | cur_dilated
    return union.transpose(2,0,1)

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
def val_video(model, val_loader, device, log_iter, val_error_dict, do_postprocessing=False, use_trimap=True, callback=None, use_temp=False):
    
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
        all_input_masks = []
        all_image_names = []
        mem_feats = None

        for i, batch in enumerate(val_loader):

            data_time.update(time.time() - end_time)

            image_names = batch.pop('image_names')
            alpha_names = None
            if 'alpha_names' in batch:
                alpha_names = batch.pop('alpha_names')
            
            transform_info = batch.pop('transform_info')
            trimap = batch.pop('trimap').numpy()
            alpha_gt = batch.pop('alpha').numpy()
            
            # Reset if new video
            if image_names[0][0].split('/')[-2] != video_name:

                if len(all_preds) > 0:
                    
                    get_single_video_metrics(callback, all_image_names, all_preds, transform_info, val_error_dict, all_trimap, all_gts, video_name, device)

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
            # alpha = postprocess(alpha)

            # cv2.imwrite("test_alpha.png", alpha[0,2,0] * 255)
            # import pdb; pdb.set_trace()

            # Fuse results
            # If no previous results, use the two first results
            if len(all_preds) == 0:
                all_preds = alpha[0]
                all_gts = alpha_gt[0]
                all_trimap = trimap[0]
                # import pdb; pdb.set_trace()
                # all_input_masks = batch['mask'].cpu().numpy()[0]
                all_image_names = image_names
                   
            else:
                # Else use the previous result and update the middle one
                all_gts = np.concatenate([all_gts, alpha_gt[0, 2:]], axis=0)
                all_trimap = np.concatenate([all_trimap, trimap[0, 2:]], axis=0)
                all_image_names += image_names[2:]
                
                
                # Fuse
                if 'diff_pred_forward' in output:
                    prev_pred = all_preds[-2]
                    next_pred = alpha[0, -1]
                    diff_forward = output['diff_pred_forward']
                    diff_backward = output['diff_pred_backward']
                    diff_forward = reverse_transform_tensor(diff_forward, transform_info).cpu().numpy()
                    diff_backward = reverse_transform_tensor(diff_backward, transform_info).cpu().numpy()
                
                    # Hard fusion
                    diff_forward = (diff_forward > 0.5).astype('float32')
                    diff_backward = (diff_backward > 0.5).astype('float32')

                    # Separate mask for each instance

                    # 2. Intersection of diff forward and union
                    # diff_forward[0, 1] = diff_forward[0, 1] * gen_roi_mask(prev_pred, alpha[0, 1])
                    # diff_backward[0, 1] = diff_backward[0, 1] * gen_roi_mask(next_pred, alpha[0, 1])

                    pred_forward01 = prev_pred * (1 - diff_forward[0, 1]) + alpha[0, 1] * diff_forward[0, 1]
                    pred_backward21 = next_pred * (1 - diff_backward[0, 1]) + alpha[0, 1] * diff_backward[0, 1]

                    # TODO: Check the diff --> update the diff forward --> fused pred based on diff forward
                    diff = np.abs(pred_forward01 - pred_backward21)
                    pred_forward01[diff > 0.0] = alpha[0, 1][diff > 0.0]
                    fused_pred = pred_forward01

                    # fused_pred = (pred_forward01 + pred_backward21) / 2.0
                    # fused_pred = pred_forward01
                    # if os.path.basename(image_names[1][0]) == '00016.jpg':
                    #     import pdb; pdb.set_trace()
                    # Ignore large difference in fusion
                    # diff = np.abs(fused_pred - alpha[0, 1])
                    # fused_pred[diff > 0.05] = alpha[0, 1][diff > 0.05]

                    # diff_forward[0, 2] = diff_forward[0, 2] * gen_roi_mask(fused_pred, alpha[0, 2])

                    # For last frame
                    pred_forward12 = fused_pred * (1 - diff_forward[0, 2]) + alpha[0, 2] * diff_forward[0, 2]

                    # if use hard fusion
                    all_preds[-1] = fused_pred
                    all_preds = np.concatenate([all_preds, pred_forward12[None]], axis=0)

                    # if not using fusion
                    # all_preds[-1] = alpha[0, 1]
                    # all_preds = np.concatenate([all_preds, alpha[0, 2][None]], axis=0)
                else:
                    all_preds = np.concatenate([all_preds, alpha[0, 2:]], axis=0)

            # Add first frame features to mem_feat
            if mem_feats is None and 'mem_feat' in output:
                # mem_feats = output['mem_feat'][None, 0:1]
                if isinstance(output['mem_feat'], tuple):
                    mem_feats = tuple(x[:, 0] for x in output['mem_feat'])
                # mem_feats = output['mem_feat'][:, 0]
            # else:
            #     mem_feats = torch.cat([mem_feats, output['mem_feat'][None, 0:1]], axis=1)
            
            # Keep at most 2 frames in mem_feat
            # if mem_feats.shape[1] > 2:
            #     mem_feats = mem_feats[:, -2:]

            if i == len(val_loader) - 1:
                get_single_video_metrics(callback, all_image_names, all_preds, transform_info, val_error_dict, all_trimap, all_gts, video_name, device)

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
    val_fn = val_video if cfg.dataset.test.name == 'MultiInstVideo' else val_image
    batch_time, data_time = val_fn(model, val_loader, device, cfg.test.log_iter, \
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


