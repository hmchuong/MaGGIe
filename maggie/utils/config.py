from yacs.config import CfgNode as CN

CONFIG = CN()

# ------------------------ General ------------------------
CONFIG.output_dir = 'logs'
CONFIG.name = 'default'

# ------------------------ Training ------------------------
CONFIG.train = CN({})
CONFIG.train.seed = -1
CONFIG.train.batch_size = 2
CONFIG.train.num_workers = 16
CONFIG.train.resume = '' # Resume from a checkpoint
CONFIG.train.resume_last = False # Resume last model or not
CONFIG.train.max_iter = 100000
CONFIG.train.log_iter = 50
CONFIG.train.vis_iter = 500
CONFIG.train.val_iter = 2000
CONFIG.train.val_metrics = ['MAD', 'MSE', 'dtSSD']
CONFIG.train.val_best_metric = 'MAD' # Metric to save the best model
CONFIG.train.val_dist = True # Evaluate distributed

optimizer = CN({})
optimizer.name = 'sgd' # sgd
optimizer.lr = 1.0e-4
optimizer.momentum = 0.9
optimizer.weight_decay = 1.0e-2
optimizer.betas = (0.9, 0.999)
CONFIG.train.optimizer = optimizer

scheduler = CN({})
scheduler.name = 'poly' # step, cosine
scheduler.power = 0.9 # for poly
scheduler.step_size = 10000 # for step
scheduler.gamma = 0.1 # for step or warmup
scheduler.warmup_iters = 1000
CONFIG.train.scheduler = scheduler

CONFIG.wandb = CN({})
CONFIG.wandb.project = 'maggie'
CONFIG.wandb.entity = 'research'
CONFIG.wandb.use = True
CONFIG.wandb.id = ''

# ------------------------ Testing ------------------------
CONFIG.test = CN({})
CONFIG.test.batch_size = 1 # Only support 1 for now
CONFIG.test.num_workers = 4
CONFIG.test.save_results = True
CONFIG.test.save_dir = 'logs'
CONFIG.test.postprocessing = True
CONFIG.test.metrics = ['MAD', 'MSE', 'SAD', 'Conn', 'Grad', 'dtSSD', 'MESSDdt']
CONFIG.test.log_iter = 50
CONFIG.test.use_trimap = True
CONFIG.test.temp_aggre = False

# ------------------------ Model ------------------------
CONFIG.model = CN({})
CONFIG.model.weights = ''
CONFIG.model.arch = 'MaGGIe'
CONFIG.model.sync_bn = True
CONFIG.model.having_unused_params = False
CONFIG.model.warmup_iters = 5000

# Encoder
CONFIG.model.encoder = 'res_encoder_29' # resnet34
CONFIG.model.encoder_args = CN({}, new_allowed=True)
CONFIG.model.encoder_args.pretrained = True
CONFIG.model.encoder_args.num_mask = 1

# ASPP
CONFIG.model.aspp = CN({})
CONFIG.model.aspp.in_channels = 512
CONFIG.model.aspp.out_channels = 512

# Decoder
CONFIG.model.decoder = ''
CONFIG.model.decoder_args = CN({}, new_allowed=True)

# For loss
CONFIG.model.loss_alpha_w = 1.0
CONFIG.model.loss_alpha_type = 'l1'
CONFIG.model.loss_alpha_grad_w = 1.0
CONFIG.model.loss_alpha_lap_w = 1.0
CONFIG.model.loss_atten_w = 1.0
CONFIG.model.loss_reweight_os8 = True
CONFIG.model.loss_dtSSD_w = 1.0

# For SHM
CONFIG.model.shm = CN({})
CONFIG.model.shm.lr_scale = 0.5
CONFIG.model.shm.dilation_kernel = 15
CONFIG.model.shm.max_n_pixel = 4000000
CONFIG.model.shm.mgm_weights = ''

refinement = CN({})
refinement.n_train_points = 2048 # number of training points for each instance
refinement.n_test_points = 16000 # maximum number of testing points for each instance
CONFIG.model.refinement = refinement

# ------------------------ Dataset ------------------------
dataset = CN({})

dataset.train = CN({})
dataset.train.name = 'VideoMatte240K'
dataset.train.root_dir = ''
dataset.train.split = 'train'
dataset.train.clip_length = 8
dataset.train.short_size = 768

# For augmentation
dataset.train.random_state = 2023
dataset.train.crop = [512, 512] # (h, w)
dataset.train.max_inst = 10
dataset.train.padding_crop_p = 0.1
dataset.train.flip_p = 0.5
dataset.train.gamma_p = 0.3

dataset.train.add_noise_p = 0.3
dataset.train.jpeg_p = 0.1
dataset.train.affine_p = 0.1
dataset.train.binarized_kernel = 30
dataset.train.downscale_mask_p = 0.5
dataset.train.mask_dir_name = "masks_matched"
dataset.train.pha_dir = 'pha'

# For video augmentation
dataset.train.max_step_size = 2
dataset.train.motion_p = 0.3

dataset.test = CN({})
dataset.test.name = 'VideoMatte240K'
dataset.test.root_dir = ''
dataset.test.split = 'valid'
dataset.test.short_size = 768
dataset.test.use_thresh_mask = False
dataset.test.downscale_mask = True
dataset.test.alpha_dir_name = "alphas"
dataset.test.mask_dir_name = "masks_matched"

# For video size
dataset.test.clip_length = 8
dataset.test.clip_overlap = 2

CONFIG.dataset = dataset