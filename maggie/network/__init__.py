import os
import logging
from .arch import *

def build_model(cfg):
    is_from_hf = False
    if cfg.weights != '' and not os.path.exists(cfg.weights):
        try:
            model = eval(cfg.arch).from_pretrained(cfg.weights)
            logging.info(f"Load pretrained model {cfg.weights} from Hugging Face")
            is_from_hf = True
        except:
            pass
    else:
        model = eval(cfg.arch)(cfg)
    return model, is_from_hf