from .vm2m_1 import VM2M
from .backbone import *

def build_model(cfg):
    backbone = eval(cfg.backbone)()
    model = eval(cfg.arch)(backbone, cfg)
    return model