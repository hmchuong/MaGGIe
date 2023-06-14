
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

from skimage.measure import label

def reverse_transform(img, transform_info):
    for transform in transform_info[::-1]:
        name = transform['name'][0]
        if name == 'padding':
            pad_h, pad_w  = transform['pad_size']
            pad_h, pad_w = pad_h.item(), pad_w.item()
            h, w = img.shape[:2]
            img = img[:h-pad_h, :w-pad_w]
        elif name == 'resize':
            h, w = transform['ori_size']
            h, w = h.item(), w.item()
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img

def postprocess(alpha, orih=None, oriw=None, bbox=None):
    labels=label((alpha>0.05).astype(int))
    try:
        assert( labels.max() != 0 )
    except:
        return None
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    alpha = alpha * largestCC
    if bbox is None:
        return alpha
    else:
        ori_alpha = np.zeros(shape=[orih, oriw], dtype=np.float32)
        ori_alpha[bbox[0]:bbox[1], bbox[2]:bbox[3]] = alpha
        return ori_alpha
    
@torch.no_grad()
def test(cfg):

    # Create dataset
    logging.info("Creating testing dataset...")
    val_dataset = build_dataset(cfg.dataset.test, is_train=False)
    val_loader = torch_data.DataLoader(
        val_dataset, batch_size=cfg.test.batch_size, shuffle=False, pin_memory=True,
        num_workers=cfg.test.num_workers)
    
    device = "cuda:0"

    # Build model
    logging.info("Building model...")
    model = build_model(cfg.model)
    model = model.to(device)

    # Load pretrained model
    assert os.path.isfile(cfg.model.weights), "Cannot find pretrained model at {}".format(cfg.model.weights)
    logging.info("Loading pretrained model from {}".format(cfg.model.weights))
    state_dict = torch.load(cfg.model.weights, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        logging.warn("Missing keys: {}".format(missing_keys))
        logging.warn("Unexpected keys: {}".format(unexpected_keys))
    
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    val_error_dict = {
        'sad': AverageMeter(),
        'mse': AverageMeter()
    }
    end_time = time.time()

    # Start training
    logging.info("Start testing...")
    model.eval()
        
    for i, batch in enumerate(val_loader):

        data_time.update(time.time() - end_time)

        image_names = batch['image_names']
        del batch['image_names']
        transform_info = batch['transform_info']
        del batch['transform_info']
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(batch)

        batch_time.update(time.time() - end_time)

        alpha = output['refined_masks']
        b, n_f, n_i = alpha.shape[:3]
        sad = torch.abs(alpha - batch['alpha']).sum() / (b * n_f * n_i) 
        mse = torch.pow(alpha - batch['alpha'], 2).sum() / (b * n_f * n_i)
        val_error_dict['sad'].update(sad.item() / 1000)
        val_error_dict['mse'].update(mse.item() / 1000)

        # Logging
        if i % cfg.test.log_iter == 0:
            logging.info("Validation: Iter {}/{}: SAD: {:.4f}, MSE: {:.4f}".format(
                    i, len(val_loader), val_error_dict['sad'].avg, val_error_dict['mse'].avg))
        
        # Visualization
        if cfg.test.save_results:
            for idx in range(len(image_names)):
                image_name = image_names[idx][0]
                video_name, image_name = image_name.split('/')[-2:]

                # Save alpha pred
                alpha_pred = (output['refined_masks'][0,idx,0] * 255).detach().cpu().numpy().astype('uint8')
                alpha_pred = postprocess(alpha_pred)
                alpha_pred = reverse_transform(alpha_pred, transform_info)

                alpha_pred_path = os.path.join(cfg.test.save_dir, video_name, 'alpha_pred')
                os.makedirs(alpha_pred_path, exist_ok=True)
                if not os.path.isfile(os.path.join(alpha_pred_path, image_name)):
                    cv2.imwrite(os.path.join(alpha_pred_path, image_name), alpha_pred)

                # Save trans pred
                trans_pred_path = os.path.join(cfg.test.save_dir, video_name, 'trans_pred')
                os.makedirs(trans_pred_path, exist_ok=True)
                trans_pred = (output['trans_preds'][0][0,idx,0].sigmoid() * 255).detach().cpu().numpy().astype('uint8')
                trans_pred = reverse_transform(trans_pred, transform_info)
                if not os.path.isfile(os.path.join(trans_pred_path, image_name)):
                    cv2.imwrite(os.path.join(trans_pred_path, image_name), trans_pred)

                for i, inc_bin_map in enumerate(output['inc_bin_maps']):
                    inc_bin_map = (inc_bin_map[0,0,0] * 255).detach().cpu().numpy().astype('uint8')
                inc_bin_path = os.path.join(cfg.test.save_dir, video_name, 'inc_pred')
                os.makedirs(inc_bin_path, exist_ok=True)
                inc_bin_pred = (output['inc_bin_maps'][0][0,0,0] * 255).detach().cpu().numpy().astype('uint8')
                inc_bin_pred = reverse_transform(inc_bin_pred, transform_info)
                if not os.path.isfile(os.path.join(inc_bin_path, image_name)):
                    cv2.imwrite(os.path.join(inc_bin_path, image_name), inc_bin_pred)

        end_time = time.time()
    
    logging.info("Validation: SAD: {:.4f}, MSE: {:.4f}".format(val_error_dict['sad'].avg, val_error_dict['mse'].avg))
    logging.info ('batch_time: {:.4f}, data_time: {:.4f}'.format(batch_time.avg, data_time.avg))


