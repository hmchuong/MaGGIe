from .vm2m_1 import VM2M
from .mgm import MGM
from .sparse_mat import SparseMat
from .tcvom import TCVOM
from .backbone import *
from .vm2m_0711 import VM2M0711

def build_model(cfg):
    backbone = eval(cfg.backbone)(**cfg.backbone_args)
    model = eval(cfg.arch)(backbone, cfg)
    return model