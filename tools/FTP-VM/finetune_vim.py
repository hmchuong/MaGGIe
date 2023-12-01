# import cv2
# cv2.setNumThreads(0)

import datetime
import time
from os import path
import math

import random
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from model.model import PropagationModel
from dataset.youtubevis import *
from dataset.youtubevos import *
from dataset.imagematte import *
from dataset.videomatte import *
from dataset.vim_dataset import *
from dataset.augmentation import *

from util.logger import TensorboardLogger
from util.hyper_para import HyperParameters


"""
Initial setup
"""
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)


# Parse command line arguments
para = HyperParameters()
para.parse()

if para['benchmark']:
    torch.backends.cudnn.benchmark = True

"""
Model related
"""

long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), para['id'])
logger = TensorboardLogger(para['id'], long_id)
logger.log_string('hyperpara', str(para))

# Construct the rank 0 model
model = PropagationModel(para, logger=logger, 
                save_path=path.join('saves', long_id, long_id) if long_id is not None else None).train()

model.PNet.load_state_dict(torch.load("saves/ftpvm.pth"))
# import pdb; pdb.set_trace()
# Automatically search the latest model path
def search_checkpoint(save_path, target):
    save_models = [
        (i, datetime.datetime.strptime(split[0], '%b%d_%H.%M.%S')) 
        for i in os.listdir(save_path) 
        if (target in i) and ((split := i.split("_"+target, maxsplit=1))[1] == '')
    ]
    
    save_models = sorted(save_models, key=lambda x: x[1], reverse=True)
    for s, t in save_models:
        p = os.path.join(save_path, s)
        ckpt_name = s+"_checkpoint.pth"
        if ckpt_name in os.listdir(p):
            ckpt_name = os.path.join(p, ckpt_name)
            return ckpt_name
    return None

# Load pertrained model if needed
seg_count = (1 - para['seg_start']) # delay start
if seg_count < 0:
    seg_count += para['seg_cd']
seg_iter = 0 if para['seg_start'] == 0 else 1e5 # 0 for seg initially, 1e5 for delay

if (model_path := para['load_model']) is not None or para['resume']:
    if para['resume']:
        print('Search model: ', para['id'])
        model_path = search_checkpoint('saves', para['id'])
        assert model_path is not None, 'No last model checkpoint exists!'
        print("Latest model ckpt: ", model_path)
    
    total_iter, extra_dict = model.load_model(model_path, ['seg_count', 'seg_iter'])
    print('Previously trained model loaded!')
    if extra_dict['seg_count'] is not None:
        seg_count = extra_dict['seg_count']
    if extra_dict['seg_iter'] is not None:
        seg_iter = extra_dict['seg_iter']
else:
    total_iter = 0

print('seg_count: %d, seg_iter: %d'%(seg_count, seg_iter))

if para['load_network'] is not None:
    model.load_network(para['load_network'])
    print('Previously trained network loaded!')

"""
Dataloader related
"""
def construct_loader(dataset, batch_size):
    train_loader = DataLoader(dataset, batch_size, num_workers=para['num_worker'],
                                shuffle=True, drop_last=True, pin_memory=True)
    return train_loader

def renew_vim_loader(nb_frame_only=False):
    size=para['size']
    seq_len=para['seq_len_video_matte']
    batch_size=para['batch_size_video_matte']
    train_dataset = VIMDataset(
        '/mnt/localssd/syn',
        None,
        None,
        size=size,
        seq_length=seq_len,
        seq_sampler=TrainFrameSampler() if nb_frame_only else TrainFrameSamplerAddFarFrame(),
        transform=VideoMatteTrainAugmentation(size, get_bgr_pha=False),
        is_VM108=True,
        bg_num=1,
        get_bgr_phas=False,
        random_memtrimap=para['random_memtrimap'],
    )
    print(f'VIM dataset size: {len(train_dataset)}, batch size: {batch_size}, sequence length: {seq_len}, frame size: {size}')
    return construct_loader(train_dataset, batch_size=batch_size)

"""
Dataset related
"""
vim_loader = renew_vim_loader(nb_frame_only=para['nb_frame_only'])
train_loader = vim_loader

"""
Roughly determine current/max epoch, 
training progress is based on iteration instead of epoch
"""
iter_base = min(len(vim_loader), para['seg_iter'])
total_epoch = math.ceil(para['iterations']/iter_base)
current_epoch = total_iter // iter_base - 1

"""
Starts training
"""
np.random.seed(np.random.randint(2**30-1))

def get_extra_dict():
    return {
        'seg_iter': seg_iter,
        'seg_count': seg_count,
    }

try:
    for e in range(current_epoch, total_epoch): 
        torch.cuda.empty_cache()
        time.sleep(2)
        if total_iter >= para['iterations']:
            break
        print('Epoch %d/%d' % (e, total_epoch))

        # Train loop
        model.train()
    
        for data in train_loader:
            model.do_pass(data, total_iter)
            total_iter += 1
            seg_count += 1

            if total_iter >= para['iterations']:
                break
                    
finally:
    if not para['debug'] and model.logger is not None and total_iter>5000:
        model.save(total_iter, ckpt_dict=get_extra_dict())
