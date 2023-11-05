from .resnet_dec import res_shortcut_decoder_22
from .resnet_fam import res_shortcut_decoder_fam_22
from .resnet_dk_dec import res_shortcut_dk_decoder_22
from .prsmha_dec import progressive_sparse_matting_dec
from .resnet_sftm import res_shortcut_sftm_22
from .resnet_atten_dec import res_shortcut_attention_decoder_22
from .resnet_dyn_atten_dec import res_shortcut_dyn_attention_decoder_22
from .resnet_embed_atten_dec import res_shortcut_embed_attention_decoder_22, res_shortcut_embed_attention_proma_decoder_22, res_shortcut_temp_embed_atten_decoder_22
from .resnet_idembed_dec import res_shortcut_id_embed_decoder_22
from .resnet_atten_spconv_dec_2 import res_shortcut_attention_spconv_decoder_22, \
    res_shortcut_attention_spconv_temp_decoder_22, \
        res_shortcut_attention_spconv_querytemp_decoder_22, \
            res_shortcut_attention_spconv_inconsisttemp_decoder_22, \
                res_shortcut_attention_spconv_bitempspar_decoder_22
# from .resnet_atten_spconv_mem_dec import res_shortcut_attention_spconv_decoder_22, res_shortcut_attention_spconv_temp_decoder_22
from .resnet_atten_spconv_lap_dec import res_shortcut_attention_spconv_lap_decoder_22
from .shm import shm

from .resnet_atten_spconv_dec_3 import res_shortcut_attention_spconv_decoder_22 as res_shortcut_attention_spconv_decoder_22_new, \
    res_shortcut_attention_spconv_bitempspar_decoder_22 as res_shortcut_attention_spconv_temp_decoder_22