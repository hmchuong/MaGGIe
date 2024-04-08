import numpy as np

import torch
from torch.nn import functional as F

from .maggie import MaGGIe

class MaGGIe_Temp(MaGGIe):

    def transform_output(self, b, n_f, h, w, n_i, pred, alpha_pred):
        output = super().transform_output(b, n_f, h, w, n_i, pred, alpha_pred)
        diff_pred_forward = pred.pop('diff_forward', None)
        diff_pred_backward = pred.pop('diff_backward', None)
        temp_alpha = pred.pop('temp_alpha', None)

        if diff_pred_backward is not None:
            
            # Adding diff_pred and temp_alpha for visualization
            diff_pred_backward = diff_pred_backward.repeat(1, 1, n_i, 1, 1)
            diff_pred_forward = diff_pred_forward.repeat(1, 1, n_i, 1, 1)
            output['diff_pred_backward'] = diff_pred_backward
            output['diff_pred_forward'] = diff_pred_forward
            output['temp_alpha'] = temp_alpha
        return output
    
    def update_additional_decoder_loss(self, pred, loss_dict):
        super().update_additional_decoder_loss(pred, loss_dict)
        if 'loss_temp' in pred:
            loss_dict['loss_temp_bce'] = pred['loss_temp_bce']
            loss_dict['loss_temp'] = pred['loss_temp']
            loss_dict['total'] += pred['loss_temp']
        if 'loss_temp_fusion' in pred:
            loss_dict['loss_temp_fusion'] = pred['loss_temp_fusion']
        if 'loss_temp_dtssd' in pred:
            loss_dict['loss_temp_dtssd'] = pred['loss_temp_dtssd']

    def forward(self, batch, **kwargs):
        output = super().forward(batch, **kwargs)

        if not self.training:
            # TODO: Post-processing
            pass