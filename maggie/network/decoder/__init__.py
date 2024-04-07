from .resnet import res_shortcut_22 # Standard MGM decoder
from .resnet_fam import res_shortcut_fam_22 # MGM + TCVOM decoder
from .resnet_inst_matt import res_shortcut_inst_matt_22 # MGM + IMD
from .resnet_inst_matt_spconv import res_shortcut_inst_matt_spconv_22, res_shortcut_inst_matt_spconv_temp_22 # MaGGIe: IMD + Spconv
from .shm import shm # SparseMat