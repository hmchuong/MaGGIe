import cv2
import torch
import numpy as np

def gen_diff_mask(alphas, k_size=25, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (k_size, k_size))
    all_diff_map = []
    for x in alphas:
        dilated = cv2.dilate(x[0, :, :, None].numpy(), kernel, iterations=iterations)
        all_diff_map.append(torch.from_numpy(dilated))
    all_diff_map = torch.stack(all_diff_map).unsqueeze(1)
    return all_diff_map

def gen_transition_gt(alphas, masks=None, k_size=25, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (k_size, k_size))
    all_trans_map = []
    for x in alphas:
        dilated = cv2.dilate(x[0, :, :, None].numpy(), kernel, iterations=iterations)
        eroded = cv2.erode(x[0, :, :, None].numpy(), kernel, iterations=iterations)
        trans_map = ((dilated - eroded) > 0).astype(float)
        all_trans_map.append(torch.from_numpy(trans_map))
    all_trans_map = torch.stack(all_trans_map).unsqueeze(1)
    
    if masks is not None:
        if masks.shape[-1] != alphas.shape[-1]:
            masks = torch.repeat_interleave(masks, 8, dim=-1)
            masks = torch.repeat_interleave(masks, 8, dim=-2)
        # upmasks = torch.repeat_interleave(masks, 8, dim=-1)
        # upmasks = torch.repeat_interleave(upmasks, 8, dim=-2)
        diff = (alphas > 127) != (masks == 255)
        all_trans_map[diff > 0] = 1.0
    
    return all_trans_map

def gen_transition_temporal_gt(alphas, masks=None, k_size=25, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (k_size, k_size))
    all_trans_map = []
    temporal_sparsity = alphas[1:] - alphas[:-1]
    temporal_sparsity = (temporal_sparsity > 1.0/255.0).float()
    for i, x in enumerate(alphas):
        dilated = cv2.dilate(x[0, :, :, None].numpy(), kernel, iterations=iterations)
        eroded = cv2.erode(x[0, :, :, None].numpy(), kernel, iterations=iterations)
        trans_map = ((dilated - eroded) > 0).astype(float)
        trans_map = torch.from_numpy(trans_map)
        if i > 0:
            trans_map[temporal_sparsity[i-1, 0] == 0] = 0.0
        all_trans_map.append(trans_map)
    all_trans_map = torch.stack(all_trans_map).unsqueeze(1)
    
    if masks is not None:
        upmasks = torch.repeat_interleave(masks, 8, dim=-1)
        upmasks = torch.repeat_interleave(upmasks, 8, dim=-2)
        diff = (alphas > 127) != (upmasks == 255)
        all_trans_map[diff > 0] = 1.0
    
    return all_trans_map

def channel_shift(xs, intensity, channel_axis):
    ys = []
    for x in xs:
        if x.ndim == 3: # image
            x = np.rollaxis(x, channel_axis, 0)
            min_x, max_x = np.min(x), np.max(x)
            channel_images = [np.clip(x_channel + intensity, min_x, max_x)
                            for x_channel in x]
            x = np.stack(channel_images, axis=0)
            x = np.rollaxis(x, 0, channel_axis + 1)
            ys.append(x)

        else:
            ys.append(x)

    return ys

def apply_transforms_cv(xs, M):
    """Apply the image transformation specified by a matrix.
    """
    dsize = (np.int32(xs[0].shape[1]), np.int32(xs[0].shape[0]))

    aff = M[:2, :2]
    off = M[:2, 2]
    cvM = np.zeros_like(M[:2, :])
    # cvM[:2,:2] = aff
    cvM[:2,:2] = np.flipud(np.fliplr(aff))
    # cvM[:2,:2] = np.transpose(aff)
    cvM[:2, 2] = np.flip(off, axis=0)
    ys = []
    for x in xs:
        if x.ndim == 3: # image
            x = cv2.warpAffine(x, cvM, dsize, flags=cv2.INTER_LINEAR)
            ys.append(x)

        else: # mask
            x = cv2.warpAffine(x, cvM, dsize, flags=cv2.INTER_NEAREST)
            ys.append(x)

    
    return ys

def flip_axis(xs, axis):
    ys = []
    for x in xs:
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        ys.append(x)

    return ys

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def random_transform(xs, rnd,
                     rt=False, # rotation
                     hs=False, # height_shift
                     ws=False, # width_shift
                     sh=False, # shear
                     zm=[1,1], # zoom
                     sc=[1,1],
                     cs=False, # channel shift
                     hf=False): # horizontal flip
                    
    """Randomly augment a single image tensor.
    """
    # x is a single image, so it doesn't have image number at index 0
    img_row_axis = 0
    img_col_axis = 1
    img_channel_axis = 2
    h, w = xs[0].shape[img_row_axis], xs[0].shape[img_col_axis]

    # use composition of homographies
    # to generate final transform that needs to be applied
    if rt:
        theta = np.pi / 180 * rnd.uniform(-rt, rt)
    else:
        theta = 0

    if hs:
        tx = rnd.uniform(-hs, hs) * h
    else:
        tx = 0 
 
    if ws:
        ty = rnd.uniform(-ws, ws) * w
    else:
        ty = 0

    if sh:
        shear = np.pi / 180 * rnd.uniform(-sh, sh)
    else:
        shear = 0

    if zm[0] == 1 and zm[1] == 1:
        zx, zy = 1, 1
    else:
        zx = rnd.uniform(zm[0], zm[1])
        zy = rnd.uniform(zm[0], zm[1])

    if sc[0] == 1 and sc[1] == 1:
        zx, zy = zx, zy
    else:
        s = rnd.uniform(sc[0], sc[1])
        zx = zx * s
        zy = zy * s

    transform_matrix = None
    if theta != 0:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix


    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                    [0, 1, ty],
                                    [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        if rnd.random() < 0.5:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
        else:
            shear_matrix = np.array([[np.cos(shear), 0, 0],
                                    [np.sin(shear), 1, 0],
                                    [0, 0, 1]])
        transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

        xs = apply_transforms_cv(xs, transform_matrix)


    if cs != 0:
        intensity = rnd.uniform(-cs, cs)
        xs = channel_shift(xs,
                            intensity,
                            img_channel_axis)
    
    if hf:
        if rnd.rand() < 0.5:
            xs = flip_axis(xs, img_col_axis)

    return xs