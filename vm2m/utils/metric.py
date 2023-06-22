import cv2
import numpy as np

def reshape2D(x):
    return x.reshape(-1, *x.shape[-2:])

class Metric(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.score = 0
        self.count = 0

    def reshape(self, pred, gt):
        if sum(pred.shape) != sum(gt.shape):
            pred = cv2.resize(pred, (gt.shape[-2], gt.shape[-1]), interpolation=cv2.INTER_LINEAR)
        return pred, gt

    def compute_metric(self, pred, gt, **kargs):
        raise NotImplementedError
    
    def update(self, pred, gt, **kargs):
        pred = reshape2D(pred)
        gt = reshape2D(gt)
        pred, gt = self.reshape(pred, gt)
        n_frames = pred.shape[0]
        self.count += n_frames
        metric = self.compute_metric(pred, gt, **kargs)
        self.score += metric
        return metric * 1.0 / (n_frames + 1e-4)

    def average(self):
        return self.score / (self.count + 1e-6)

class SAD(Metric):
    
    def compute_metric(self, pred, gt, **kargs):
        '''
        pred, gt: numpy array
        (N, *, H, W)
        '''
        return np.sum(np.abs(pred - gt)) / 1000.0

class MSE(Metric):
    
    def compute_metric(self, pred, gt, **kargs):
        '''
        pred, gt: numpy array
        (N, *, H, W)
        '''
        n_pixels = pred.shape[1] * pred.shape[2]
        return (np.sum((pred - gt) ** 2) / n_pixels)

class MAD(Metric):

    def compute_metric(self, pred, gt, **kargs):
        n_pixels = pred.shape[1] * pred.shape[2]
        return (np.sum(np.abs(pred - gt)) / n_pixels)

class MaskedMAD(Metric):
    def compute_metric(self, pred, gt, mask, **kargs):
        
        mask = reshape2D(mask)
        n_pixels = mask.sum((-1, -2))
        # import pdb; pdb.set_trace()
        return (np.sum(np.abs(pred - gt) * mask, (-1, -2)) / n_pixels).sum()

class FgMAD(MaskedMAD):
    def compute_metric(self, pred, gt, trimap, **kargs):
        fg_mask = (trimap == 2)
        return super().compute_metric(pred, gt, fg_mask, **kargs)

class BgMAD(MaskedMAD):
    def compute_metric(self, pred, gt, trimap, **kargs):
        bg_mask = (trimap == 0)
        return super().compute_metric(pred, gt, bg_mask, **kargs)

class TransMAD(MaskedMAD):
    def compute_metric(self, pred, gt, trimap, **kargs):
        trans_mask = (trimap == 1)
        return super().compute_metric(pred, gt, trans_mask, **kargs)
    
class Conn(Metric):
    
    def compute_metric(self, pred, gt, **kargs):
        conn_err = 0
        for i in range(pred.shape[0]):
            conn_err += self.compute_conn(pred[i], gt[i])
        return conn_err / 1000.0

    def compute_conn(self, pred, gt, step=0.1):
        """
        update metric.
        Args:
            pred (np.ndarray): The value range is [0., 1.].
            gt (np.ndarray): The value range is [0, 1].
            step (float, optional): Step of threshold when computing intersection between
            `gt` and `pred`. Default: 0.1.
        """

        thresh_steps = np.arange(0, 1 + step, step)
        round_down_map = -np.ones_like(gt)
        for i in range(1, len(thresh_steps)):
            gt_thresh = gt >= thresh_steps[i]
            pred_thresh = pred >= thresh_steps[i]
            intersection = (gt_thresh & pred_thresh).astype(np.uint8)

            # connected components
            _, output, stats, _ = cv2.connectedComponentsWithStats(intersection, connectivity=4)
            # start from 1 in dim 0 to exclude background
            size = stats[1:, -1]

            # largest connected component of the intersection
            omega = np.zeros_like(gt)
            if len(size) != 0:
                max_id = np.argmax(size)
                # plus one to include background
                omega[output == max_id + 1] = 1

            mask = (round_down_map == -1) & (omega == 0)
            round_down_map[mask] = thresh_steps[i - 1]
        round_down_map[round_down_map == -1] = 1

        gt_diff = gt - round_down_map
        pred_diff = pred - round_down_map
        # only calculate difference larger than or equal to 0.15
        gt_phi = 1 - gt_diff * (gt_diff >= 0.15)
        pred_phi = 1 - pred_diff * (pred_diff >= 0.15)

        conn_diff = np.sum(np.abs(gt_phi - pred_phi))
        return conn_diff

class Grad(Metric):
    
    def gaussian(self, x, sigma):
        return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    def dgaussian(self, x, sigma):
        return -x * self.gaussian(x, sigma) / sigma**2

    def gauss_filter(self, sigma, epsilon=1e-2):
        half_size = np.ceil(
            sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = int(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = self.gaussian(
                    i - half_size, sigma) * self.dgaussian(j - half_size, sigma)

        # normalize filter
        norm = np.sqrt((filter_x**2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y

    def gauss_gradient(self, img, sigma):
        filter_x, filter_y = self.gauss_filter(sigma)
        img_filtered_x = cv2.filter2D(
            img, -1, filter_x, borderType=cv2.BORDER_REPLICATE)
        img_filtered_y = cv2.filter2D(
            img, -1, filter_y, borderType=cv2.BORDER_REPLICATE)
        return np.sqrt(img_filtered_x**2 + img_filtered_y**2)
    
    def compute_grad(self, pred, gt, sigma=1.4):

        gt = gt.astype(np.float64)
        pred = pred.astype(np.float64)
        gt_normed = np.zeros_like(gt)
        pred_normed = np.zeros_like(pred)
        cv2.normalize(gt, gt_normed, 1., 0., cv2.NORM_MINMAX)
        cv2.normalize(pred, pred_normed, 1., 0., cv2.NORM_MINMAX)

        gt_grad = self.gauss_gradient(gt_normed, sigma).astype(np.float32)
        pred_grad = self.gauss_gradient(pred_normed, sigma).astype(np.float32)

        grad_diff = ((gt_grad - pred_grad)**2).sum()

        return grad_diff
    
    def compute_metric(self, pred, gt, **kargs):
        grad_err = 0
        for i in range(pred.shape[0]):
            grad_err += self.compute_grad(pred[i], gt[i])
        return grad_err / 1000.0

def build_metric(metrics):
    '''
    metrics: list of str
    returns:
    dict of metric name and metric class
    '''
    metric_dict = {}
    for metric in metrics:
        try:
            metric_dict[metric] = eval(metric)()
        except:
            raise NotImplementedError(f'metric {metric} is not implemented')
    return metric_dict