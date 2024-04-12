import numpy as np

from .maggie import MaGGIe

class MaGGIe_Temp(MaGGIe):

    def transform_output(self, b, n_f, h, w, n_i, pred, alpha_pred):
        output = super().transform_output(b, n_f, h, w, n_i, pred, alpha_pred)
        diff_pred_forward = pred.pop('diff_forward', None)
        diff_pred_backward = pred.pop('diff_backward', None)
        temp_alpha = pred.pop('temp_alpha', None) # Average between forward and backward

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
            # Post-processing by our alpha-matte level aggregation algorithm

            alphas = output["refined_masks"] # (1, 3, n_i, H, W)

            # t-1 output from previous prediction if available
            prev_pred = kwargs.get('prev_pred', alphas[:, 0]) # (1, n_i, H, W)

            # t + 1 output of current predictions
            next_pred = alphas[:, -1] # (1, n_i, H, W)

            diff_forward = output['diff_pred_forward']
            diff_backward = output['diff_pred_backward']

            # Thresholding
            diff_forward = (diff_forward > 0.5).astype('float32')
            diff_backward = (diff_backward > 0.5).astype('float32')

            # Forward propagate from t-1 to t
            pred_forward01 = prev_pred * (1 - diff_forward[:, 1]) + alphas[:, 1] * diff_forward[:, 1] # (1, n_i, H, W)

            # Backward propagate from t+1 to t
            pred_backward21 = next_pred * (1 - diff_backward[:, 1]) + alphas[:, 1] * diff_backward[:, 1] # (1, n_i, H, W)

            # Check the diff --> update the diff forward --> fused pred based on diff forward
            diff = np.abs(pred_forward01 - pred_backward21)

            # Use pred t from the model with pred forward != pred backward
            pred_forward01[diff > 0.0] = alphas[:, 1][diff > 0.0]

            # Update the middle alpha t
            alphas[:, 1] = pred_forward01

            # Update the last alpha t+1
            pred_forward12 = pred_forward01 * (1 - diff_forward[:, 2]) + next_pred * diff_forward[:, 2]
            alphas[:, 2] = pred_forward12
        
        return output


