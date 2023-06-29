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
CONFIG.train.resume = ''
CONFIG.train.max_iter = 100000
CONFIG.train.log_iter = 50
CONFIG.train.vis_iter = 500
CONFIG.train.val_iter = 2000
CONFIG.train.val_metrics = ['BgMAD', 'FgMAD', 'TransMAD', 'MAD', 'MSE']
CONFIG.train.val_best_metric = 'MAD'

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
scheduler.gamma = 0.1 # for step
scheduler.warmup_iters = 1000
CONFIG.train.scheduler = scheduler

CONFIG.wandb = CN({})
CONFIG.wandb.project = 'vm2m'
CONFIG.wandb.entity = 'chuongh'
CONFIG.wandb.use = True
CONFIG.wandb.id = ''

# ------------------------ Testing ------------------------
CONFIG.test = CN({})
CONFIG.test.batch_size = 1 # Only support 1 for now
CONFIG.test.num_workers = 16
CONFIG.test.save_results = True
CONFIG.test.save_dir = 'logs'
CONFIG.test.postprocessing = True
# CONFIG.test.metrics = ['BgMAD', 'FgMAD', 'TransMAD', 'MAD', 'SAD', 'MSE', 'Conn', 'Grad']
CONFIG.test.metrics = ['BgMAD', 'FgMAD', 'TransMAD', 'MAD', 'MSE']
CONFIG.test.log_iter = 50

# ------------------------ Model ------------------------
CONFIG.model = CN({})
CONFIG.model.weights = ''
CONFIG.model.arch = 'VM2M'
CONFIG.model.backbone = 'res_encoder_29' # resnet34

# For MGM
CONFIG.model.mgm = CN({})
CONFIG.model.mgm.decoder = 'res_shortcut_decoder_22'
CONFIG.model.mgm.warmup_iter = 5000

# For loss
CONFIG.model.loss_alpha_w = 1.0
CONFIG.model.loss_alpha_type = 'l1'
CONFIG.model.loss_alpha_grad_w = 1.0
CONFIG.model.loss_alpha_lap_w = 1.0

CONFIG.model.loss_comp_w = 0.25
CONFIG.model.loss_dtSSD_w = 1.0

# Dynamic kernel
dynamic_kernel = CN({})
dynamic_kernel.in_features = ['os32', 'os16', 'os8']
dynamic_kernel.hidden_dim = 256
dynamic_kernel.nheads = 4
dynamic_kernel.dim_feedforward = 256
dynamic_kernel.dec_layers = 5
dynamic_kernel.pre_norm = False
dynamic_kernel.enforce_input_project = True
dynamic_kernel.in_channels = 256
dynamic_kernel.out_incoherence = 157 # 99
dynamic_kernel.out_pixeldecoder = 64
CONFIG.model.dynamic_kernel = dynamic_kernel

# Breakdown incoherence
breakdown = CN({})
breakdown.in_channels = [32, 32, 32]
breakdown.in_features = ['os4', 'os1']
CONFIG.model.breakdown = breakdown

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
dataset.train.flip_prob = 0.5

# For video augmentation
dataset.train.max_step_size = 2
dataset.train.bg_dir = ''
dataset.train.blur_prob = 0.5
dataset.train.blur_kernel_size = [5, 15, 25]
dataset.train.blur_sigma = [1.0, 1.5, 3.0, 5.0]
dataset.train.bin_alpha_max_k = 30

dataset.test = CN({})
dataset.test.name = 'VideoMatte240K'
dataset.test.root_dir = ''
dataset.test.split = 'valid'

# For video augmentation
dataset.test.clip_length = 8
dataset.test.clip_overlap = 2
dataset.test.short_size = 768

CONFIG.dataset = dataset