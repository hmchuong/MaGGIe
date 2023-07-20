from .arch import *
from .backbone import *
from .decoder import *

def build_model(cfg):
    backbone = eval(cfg.backbone)(**cfg.backbone_args)
    
    if cfg.decoder == '':
        decoder = None
    else:
        decoder = eval(cfg.decoder)(**cfg.decoder_args)
    
    model = eval(cfg.arch)(backbone, decoder, cfg)
    return model