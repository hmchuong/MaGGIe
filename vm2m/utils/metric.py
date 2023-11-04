import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.measure
from .dist import synchronize, gather
from multiprocessing import Pool
from joblib import Parallel, delayed

def reshape2D(x):
    return x.reshape(-1, *x.shape[-2:])

class Metric(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.score = 0
        self.count = 0

    # def reshape(self, pred, gt):
    #     gt_shape = gt.shape
    #     if len(pred.shape) > 4:
            
    #     if sum(pred.shape) != sum(gt.shape):
    #         pred = cv2.resize(pred, (gt.shape[-2], gt.shape[-1]), interpolation=cv2.INTER_LINEAR)
    #     return pred

    def compute_metric(self, pred, gt, **kargs):
        raise NotImplementedError
    
    def gather_metric(self, rank=0):
        synchronize()
        gather_score = gather(self.score, dst=rank)
        gather_score = sum(gather_score)
        gather_count = gather(self.count, dst=rank)
        gather_count = sum(gather_count)
        self.score = gather_score
        self.count = gather_count


    def update(self, pred, gt, trimap=None, **kargs):
        
        mask = None
        if trimap is not None:
            mask = (trimap > 0).astype('float32')
        else:
            mask = np.ones_like(gt).astype('float32')

        pred = reshape2D(pred)
        gt = reshape2D(gt)
        mask = reshape2D(mask)

        # pred, gt = self.reshape(pred, gt)
        score, count = self.compute_metric(pred, gt, mask, **kargs)
        # import pdb; pdb.set_trace()
        # self.count += count
        # self.score += score
        self.count += count
        self.score += score
        return score * 1.0 / count

    def average(self):
        return self.score / (self.count + 1e-6)

class SAD(Metric):
    
    def compute_metric(self, pred, gt, mask, **kargs):
        '''
        pred, gt: numpy array
        (N, *, H, W)
        '''
        # return np.sum(np.abs(pred - gt) * mask) * 0.001, mask.shape[0]
        diff = np.abs(pred - gt) * mask
        sad = np.sum(diff, axis=(1, 2))
        return sad.sum() * 1e-3, mask.shape[0]

class MSE(Metric):
    
    def compute_metric(self, pred, gt, mask, **kargs):
        '''
        pred, gt: numpy array
        (N, *, H, W)
        '''
        # return np.sum(((pred - gt) ** 2) * mask) * 1000, mask.sum()
        diff = ((pred - gt) ** 2) * mask
        mse = np.mean(diff, axis=(1, 2)) / (mask.sum(axis=(1, 2)) + 1e-6)
        return mse.sum() * 1e10, mask.shape[0]

class MAD(Metric):

    def compute_metric(self, pred, gt, mask, **kargs):
        # return np.sum(np.abs(pred - gt) * mask) * 1000, mask.sum()
        diff = np.abs(pred - gt) * mask
        mad = np.mean(diff, axis=(1, 2)) / (mask.sum(axis=(1, 2)) + 1e-6)
        return mad.sum() * 1e10, mask.shape[0]

# class MAD_fg(Metric):
#     def compute_metric(self, pred, gt, mask, **kargs):
#         mask = (mask == 2).float()
#         return np.sum(np.abs(pred - gt) * mask) * 1000, mask.sum()

# class MAD_bg(Metric):
#     def compute_metric(self, pred, gt, mask, **kargs):
#         mask = (mask == 0).float()
#         return np.sum(np.abs(pred - gt) * mask) * 1000, mask.sum()

# class MAD_unk(Metric):
#     def compute_metric(self, pred, gt, mask, **kargs):
#         mask = (mask == 1).float()
#         return np.sum(np.abs(pred - gt) * mask) * 1000, mask.sum()

# class Conn(Metric):
    
#     def compute_metric(self, pred, gt, mask, **kargs):
#         conn_err = 0
#         B = pred.shape[0]
#         # mask = np.ones_like(mask)
#         pool = Pool(B)
#         for err in pool.imap(self.compute_conn, zip(pred, gt, mask)):
#             conn_err += err
#         # for i in range(pred.shape[0]):
#         #     conn_err += self.compute_conn((pred[i], gt[i], mask[i]))
#         pool.close()
#         # import pdb; pdb.set_trace()
#         return conn_err, B

#     def compute_conn(self, args):
#         """
#         update metric.
#         Args:
#             pred (np.ndarray): The value range is [0., 1.].
#             gt (np.ndarray): The value range is [0, 1].
#             step (float, optional): Step of threshold when computing intersection between
#             `gt` and `pred`. Default: 0.1.
#         """
#         pred, gt, roi_mask = args
#         step=0.1
#         thresh_steps = np.arange(0, 1 + step, step)
#         round_down_map = -np.ones_like(gt)
#         for i in range(1, len(thresh_steps)):
#             gt_thresh = gt >= thresh_steps[i]
#             pred_thresh = pred >= thresh_steps[i]
#             intersection = (gt_thresh & pred_thresh).astype(np.uint8)

#             # connected components
#             _, output, stats, _ = cv2.connectedComponentsWithStats(intersection, connectivity=4)
#             # start from 1 in dim 0 to exclude background
#             size = stats[1:, -1]

#             # largest connected component of the intersection
#             omega = np.zeros_like(gt)
#             if len(size) != 0:
#                 max_id = np.argmax(size)
#                 # plus one to include background
#                 omega[output == max_id + 1] = 1

#             mask = (round_down_map == -1) & (omega == 0)
#             round_down_map[mask] = thresh_steps[i - 1]
#         round_down_map[round_down_map == -1] = 1

#         gt_diff = gt - round_down_map
#         pred_diff = pred - round_down_map
#         # only calculate difference larger than or equal to 0.15
#         gt_phi = 1 - gt_diff * (gt_diff >= 0.15)
#         pred_phi = 1 - pred_diff * (pred_diff >= 0.15)
#         conn_diff = np.sum(np.abs(gt_phi - pred_phi) * roi_mask)
#         return conn_diff

# class Conn(Metric):

#     def compute_metric(self, pred, gt, mask, **kargs):
#         conn_err = 0
#         B = pred.shape[0]
#         # mask = np.ones_like(mask)
#         pool = Pool(B)
#         for err in pool.imap(self.compute_conn, zip(pred, gt, mask)):
#             conn_err += err * 0.001
#         # for i in range(pred.shape[0]):
#         #     conn_err += self.compute_conn((pred[i], gt[i], mask[i]))
#         pool.close()
#         # import pdb; pdb.set_trace()
#         return conn_err, B

#     def compute_conn(self, args):
#         """
#         update metric.
#         Args:
#             pred (np.ndarray): The value range is [0., 1.].
#             gt (np.ndarray): The value range is [0, 1].
#             step (float, optional): Step of threshold when computing intersection between
#             `gt` and `pred`. Default: 0.1.
#         """
#         pred, gt, roi_mask = args
#         step=0.1
#         thresh_steps = np.arange(0, 1 + step, step)
#         round_down_map = -np.ones_like(gt)
#         for i in range(1, len(thresh_steps)):
#             gt_thresh = gt >= thresh_steps[i]
#             pred_thresh = pred >= thresh_steps[i]
#             intersection = (gt_thresh & pred_thresh).astype(np.uint8)

#             cc, num = skimage.measure.label(intersection, connectivity=1, return_num=True)
#             omega = np.zeros_like(intersection)
#             if num > 0:
#                 # find the largest connected region
#                 max_id = np.argmax(np.bincount(cc.flatten())[1:]) + 1
#                 omega[cc == max_id] = 1

#             mask = (round_down_map == -1) & (omega == 0)
#             round_down_map[mask] = thresh_steps[i - 1]
#         round_down_map[round_down_map == -1] = 1

#         gt_diff = gt - round_down_map
#         pred_diff = pred - round_down_map
#         # only calculate difference larger than or equal to 0.15
#         gt_phi = 1 - gt_diff * (gt_diff >= 0.15)
#         pred_phi = 1 - pred_diff * (pred_diff >= 0.15)
#         conn_diff = np.sum(np.abs(gt_phi - pred_phi) * roi_mask)
#         return conn_diff

class Conn(Metric):

    def compute_metric(self, pred, gt, mask, **kargs):
        conn_err = self.compute_conn(pred, gt, mask) * 0.001
        B = pred.shape[0]
        return conn_err, B

    @staticmethod
    def compute_largest_connected_component(intersection):
        cc, num = skimage.measure.label(intersection, connectivity=1, return_num=True)
        omega = np.zeros_like(intersection)
        if num > 0:
            max_id = np.argmax(np.bincount(cc.flatten())[1:]) + 1
            omega[cc == max_id] = 1
        return omega

    def compute_conn(self, pred, gt, roi_mask):
        """
        update metric.
        Args:
            pred (np.ndarray): The value range is [0., 1.].
            gt (np.ndarray): The value range is [0, 1].
            step (float, optional): Step of threshold when computing intersection between
            `gt` and `pred`. Default: 0.1.
        """
        step=0.1
        B = pred.shape[0]
        thresh_steps = np.arange(0, 1 + step, step)
        round_down_map = -np.ones_like(gt)
        all_intersections = []
        for b in range(B):
            for i in range(1, len(thresh_steps)):
                gt_thresh = gt[b] >= thresh_steps[i]
                pred_thresh = pred[b] >= thresh_steps[i]
                intersection = (gt_thresh & pred_thresh).astype(np.uint8)
                all_intersections.append(intersection)

        
        # with Pool(4) as p:
        #     all_omegas = p.map(self.compute_largest_connected_component, all_intersections)
        all_omegas = Parallel(n_jobs=min(10, len(all_intersections)))(delayed(self.compute_largest_connected_component)(intersection) for intersection in all_intersections)

        j = 0
        for b in range(B):
            for i in range(1, len(thresh_steps)):
                omega = all_omegas[j]
                j += 1
                mask = (round_down_map[b] == -1) & (omega == 0)
                round_down_map[b][mask] = thresh_steps[i - 1]
        
        round_down_map[round_down_map == -1] = 1

        gt_diff = gt - round_down_map
        pred_diff = pred - round_down_map
        # only calculate difference larger than or equal to 0.15
        gt_phi = 1 - gt_diff * (gt_diff >= 0.15)
        pred_phi = 1 - pred_diff * (pred_diff >= 0.15)
        conn_diff = np.sum(np.abs(gt_phi - pred_phi) * roi_mask)
        del all_omegas, all_intersections, round_down_map, gt_diff, pred_diff, gt_phi, pred_phi
        gc.collect()
        return conn_diff

# class Grad(Metric):
    
#     def gaussian(self, x, sigma):
#         return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

#     def dgaussian(self, x, sigma):
#         return -x * self.gaussian(x, sigma) / sigma**2

#     def gauss_filter(self, sigma, epsilon=1e-2):
#         half_size = np.ceil(
#             sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
#         size = int(2 * half_size + 1)

#         # create filter in x axis
#         filter_x = np.zeros((size, size))
#         for i in range(size):
#             for j in range(size):
#                 filter_x[i, j] = self.gaussian(
#                     i - half_size, sigma) * self.dgaussian(j - half_size, sigma)

#         # normalize filter
#         norm = np.sqrt((filter_x**2).sum())
#         filter_x = filter_x / norm
#         filter_y = np.transpose(filter_x)

#         return filter_x, filter_y

#     def gauss_gradient(self, img, sigma):
#         filter_x, filter_y = self.gauss_filter(sigma)
#         img_filtered_x = cv2.filter2D(
#             img, -1, filter_x, borderType=cv2.BORDER_REPLICATE)
#         img_filtered_y = cv2.filter2D(
#             img, -1, filter_y, borderType=cv2.BORDER_REPLICATE)
#         return np.sqrt(img_filtered_x**2 + img_filtered_y**2)
    
#     def compute_grad(self, args):
#         pred, gt, mask = args
#         sigma=1.4
#         gt = gt.astype(np.float64)
#         pred = pred.astype(np.float64)
#         gt_normed = np.zeros_like(gt)
#         pred_normed = np.zeros_like(pred)
#         cv2.normalize(gt, gt_normed, 1., 0., cv2.NORM_MINMAX)
#         cv2.normalize(pred, pred_normed, 1., 0., cv2.NORM_MINMAX)

#         gt_grad = self.gauss_gradient(gt_normed, sigma).astype(np.float32)
#         pred_grad = self.gauss_gradient(pred_normed, sigma).astype(np.float32)

#         grad_diff = (((gt_grad - pred_grad)**2) * mask).sum()

#         return grad_diff
    
#     def compute_metric(self, pred, gt, mask, **kargs):
#         grad_err = 0
#         B = pred.shape[0]
#         pool = Pool(B)
#         for err in pool.imap(self.compute_grad, zip(pred, gt, mask)):
#             grad_err += err * 0.001
#         pool.close()
#         return grad_err, B

class Grad(Metric):
    def __init__(self):
        super().__init__()
        sigma = 1.4
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
        
        # Convert filters to PyTorch tensors and move to GPU
        self.filter_x = torch.tensor(self.filter_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.filter_y = torch.tensor(self.filter_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
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

    def gauss_gradient(self, img):
        img_filtered_x = F.conv2d(img, self.filter_x, padding=self.filter_x.shape[-1]//2)
        img_filtered_y = F.conv2d(img, self.filter_y, padding=self.filter_y.shape[-1]//2)
        return torch.sqrt(img_filtered_x**2 + img_filtered_y**2)
    
    def compute_grad(self, pred, gt, mask):
        gt = gt.float().unsqueeze(1) # B x 1 x H x W
        pred = pred.float().unsqueeze(1)#.cuda()
        mask = mask.float().unsqueeze(1)#.cuda()

        gt_normed = (gt - gt.min()) / (gt.max() - gt.min() + 1e-6)
        pred_normed = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)

        gt_grad = self.gauss_gradient(gt_normed)
        pred_grad = self.gauss_gradient(pred_normed)

        grad_diff = (((gt_grad - pred_grad)**2) * mask).sum().item()
        del gt_grad, pred_grad, gt_normed, pred_normed
        torch.cuda.empty_cache()
        gc.collect()
        return grad_diff
    
    def compute_metric(self, pred, gt, mask, device='cuda', **kargs):
        
        pred = torch.from_numpy(pred).to(device)
        gt = torch.from_numpy(gt).to(device)
        mask = torch.from_numpy(mask).to(device)

        self.filter_x = self.filter_x.to(device)
        self.filter_y = self.filter_y.to(device)

        grad_err = self.compute_grad(pred, gt, mask) * 0.001
        B = pred.shape[0]
        return grad_err, B

class dtSSD(Metric):

    def update(self, pred, gt, trimap=None, **kargs):
        mask = None
        if trimap is not None:
            mask = (trimap == 1).astype('float32')
        else:
            mask = np.ones_like(gt).astype('float32')

        dadt = pred[:, 1:] - pred[:, :-1]
        dgdt = gt[:, 1:] - gt[:, :-1]
        mask_0 = mask[:, :-1]
        err_m = (dadt - dgdt) ** 2
        err_m = err_m * mask_0
        err = np.sqrt(np.sum(err_m, axis=(0, 1, 3, 4)))
        err = np.sum(err) * 0.1
        num = mask_0.shape[2] #mask_0.sum()

        # dtSSD for each instance in each video

        self.score += err
        self.count += num
        return err / (num + 1e-10)
    
class MESSDdt(Metric):
    def calcOpticalFlow(self, frames):
        prev, curr = frames
        flow = cv2.calcOpticalFlowFarneback(prev.astype(np.uint8), curr.astype(np.uint8), None,  
                                        0.5, 5, 10, 2, 7, 1.5, 
                                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        return flow
    
    def compute_single_video(self, pred, gt, mask):
        pred = reshape2D(pred)
        gt = reshape2D(gt)
        
        B, h, w = gt.shape
        pool = Pool(B)
        flows = []
        items = [t for t in (gt * 255)]
        for flow in pool.imap(self.calcOpticalFlow, zip(items[:-1], items[1:])):
            flows.append(flow)
        flow = torch.from_numpy(np.rint(np.array(flows)).astype(np.int64))
        pool.close()

        pred = torch.from_numpy(pred)
        gt = torch.from_numpy(gt)
        mask = torch.from_numpy(mask)
        pred_0 = pred[:-1, ...]
        pred_1 = pred[1:, ...]
        target_0 = gt[:-1, ...]
        target_1 = gt[1:, ...]
        mask_0 = mask[:-1, ...]
        mask_1 = mask[1:, ...]
        
        B, h, w = target_0.shape
        x = torch.arange(0, w)
        y = torch.arange(0, h)
        xx, yy = torch.meshgrid([y, x])
        coords = torch.stack([yy, xx], dim=2).unsqueeze(0).repeat((B, 1, 1, 1))
        coords_n = (coords + flow)
        coords_y = coords_n[..., 0].clamp(0, h-1)
        coords_x = coords_n[..., 1].clamp(0, w-1)
        indices = coords_y * w + coords_x
        pred_1 = torch.take(pred_1, indices)
        target_1 = torch.take(target_1, indices)
        mask_1 = torch.take(mask_1, indices)

        error_map = (pred_0-target_0).pow(2) * mask_0 - (pred_1-target_1).pow(2) * mask_1

        error = error_map.abs().view(mask_0.shape[0], -1).sum(dim=1) # (N_f - 1) x HW
        num = mask_0.view(mask_0.shape[0], -1).sum(dim=1) + 1. # (N_f - 1) x HW
        
        error = error.cpu().numpy().sum() / num.cpu().numpy().sum()
        # num = num.cpu().numpy().sum()
        return error
    
    def update(self, pred, gt, trimap=None, **kargs):
        if pred.ndim == 5:
            pred = pred.squeeze(0)
            gt = gt.squeeze(0)
        mask = None
        if trimap is not None:
            mask = (trimap == 1).astype('float32')
        else:
            mask = np.ones_like(gt).astype('float32')

        error = 0
        count = 0

        # N_F x N_I x H x W


        for i in range(pred.shape[1]):
            try:
                e = self.compute_single_video(pred[:, i], gt[:, i], mask[:, i])
            except Exception as exception:
                print(exception)
                continue
            error += e * 10000
            count += 1
        # all_omegas = Parallel(n_jobs=len(pred))(delayed(self.compute_single_video)(pred[i], gt[i], mask[i]) for intersection in all_intersections)
            
        self.score += error # Sum of error for each instance
        self.count += count # Add number of instances
        return error / (count + 1e-8)
        

def build_metric(metrics):
    '''
    metrics: list of str
    returns:
    dict of metric name and metric class
    '''
    metric_dict = {}
    for metric in metrics:
        # try:
        metric_dict[metric] = eval(metric)()
        # except:
        #     raise NotImplementedError(f'metric {metric} is not implemented')
    return metric_dict