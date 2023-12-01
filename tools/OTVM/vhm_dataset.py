import glob
from dataset import *

def _flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

class VIM_Train(torch.utils.data.Dataset):
    def __init__(self, data_root, image_shape,
                 mode='train',
                 use_subset=False,
                 sample_length=3,
                 max_skip=75,
                 do_affine=0.5,
                 do_time_flip=0.5,
                 do_histogram_matching=0.3,
                 do_gamma_aug=0.3,
                 do_jpeg_aug=0.3,
                 do_gaussian_aug=0.3,
                 do_motion_aug=0.3,):
        self.mode = mode
        self.use_subset = use_subset
        self.sample_length = sample_length
        self.max_skip = max_skip
        self.do_affine = do_affine
        self.do_time_flip = do_time_flip
        self.do_histogram_matching = do_histogram_matching
        self.do_gamma_aug = do_gamma_aug
        self.do_jpeg_aug = do_jpeg_aug
        self.do_gaussian_aug = do_gaussian_aug
        self.do_motion_aug = do_motion_aug
        assert self.mode in ['train', 'val']
        self.root = data_root
        self.image_shape = list(image_shape)

        self.data_root = dict()
        self.FG = list()
        self.Alpha = list()
        
        self.data_root['VIM'] = data_root
        self.parse_vim(data_root, self.mode)
        
        self.FG_len = len(self.FG)
        
        self.pixel_aug_gamma = iaa.GammaContrast(gamma=iap.TruncatedNormal(1.0, 0.2, 0.5, 1.5))
        self.pixel_aug_gaussian = iaa.AdditiveGaussianNoise(scale=(0, 0.03*255))
        self.jpeg_aug = iaa.JpegCompression(compression=(20, 80)) 
        self.motion_aug = A.MotionBlur(p=1.0, blur_limit=(3,49))
        

        self.EdgeFilter = nn.Conv2d(1, 2, kernel_size=(3,3), stride=1, bias=False) # No Padding
        self.EdgeFilter.weight = nn.Parameter(torch.Tensor([[[[1., 0., -1.,],
                                                                [2., 0,  -2.,],
                                                                [1., 0., -1.]]],
                                                            [[[ 1.,  2.,  1.,],
                                                                [ 0.,  0,   0.,],
                                                                [-1., -2., -1.]]]]))
        for param in self.EdgeFilter.parameters():
            param.requires_grad = False
    
    def __len__(self):
        return self.FG_len
    
    def parse_vim(self, data_root, mode):
        subdir = "train" if mode == 'train' else "becnhmark/comp_medium"
        
        # Load FG, pairs for each instance
        video_names = os.listdir(os.path.join(data_root, subdir, "fgr"))
        video_names.sort()

        for video_name in video_names:
            video_path = os.path.join(data_root, subdir, "fgr", video_name)
            frame_names = os.listdir(video_path)
            frame_names.sort()
            all_alpha_paths = glob.glob(os.path.join(data_root, subdir, "pha", video_name) + "/*/*.png")
            num_instances = len(all_alpha_paths) // len(frame_names)
            FG_current = [os.path.join(video_path, frame_name) for frame_name in frame_names]
            for i_instance in range(num_instances):   
                Alpha_current = [fg_path.replace("/fgr/", "/pha/").replace(".jpg", "") + f"/{i_instance:02d}.png" for fg_path in FG_current]
                self.FG.append(['VIM', FG_current])
                self.Alpha.append(['VIM', Alpha_current])
    

    def random_crop(self, N_frames, N_masks, num_frames, rnd):
        real_size = N_frames[0].shape[:2]
        do_mask = N_masks is not None
        ## random transformations that both to be applied.
        min_scale = np.maximum(self.image_shape[0]/np.float32(real_size[0]), self.image_shape[1]/np.float32(real_size[1]))

        for t in range(100):
            scale = np.maximum(rnd.choice([1., 1./1.5, 1./2.]), min_scale+0.01)
            dsize = (np.int64(real_size[1]*scale), np.int64(real_size[0]*scale))

            _rz_N_frames = np.empty((num_frames, dsize[1], dsize[0], N_frames[0].shape[2]), dtype=np.float32)
            if do_mask:
                _rz_N_masks = np.empty((num_frames, dsize[1], dsize[0]), dtype=np.float32) 
            for f in range(num_frames):
                _rz_N_frames[f] = cv2.resize(N_frames[f], dsize=dsize, interpolation=cv2.INTER_LINEAR)
                if do_mask:
                    _rz_N_masks[f] = cv2.resize(N_masks[f], dsize=dsize, interpolation=cv2.INTER_LINEAR)
            rz_N_frames = _rz_N_frames
            if do_mask:
                rz_N_masks = _rz_N_masks
                np_in1 = None
            for tt in range(1000):
                cr_y = rnd.randint(0, _rz_N_frames.shape[1] - self.image_shape[0])
                cr_x = rnd.randint(0, _rz_N_frames.shape[2] - self.image_shape[1])
                if do_mask:
                    center_alpha_val = rz_N_masks[0, cr_y+int(self.image_shape[0]//2), cr_x+int(self.image_shape[1]//2)]
                    if (tt < 900) or (t < 90):
                        if (center_alpha_val > (0.2*255)) and (center_alpha_val < (0.8*255)):
                            crop_N_masks = rz_N_masks[:,cr_y:cr_y+self.image_shape[0], cr_x:cr_x+self.image_shape[1]]
                            break
                    else:
                        if np_in1 is None:
                            np_in1 = np.sum((rz_N_masks[0] > (0.2*255)) & (rz_N_masks[0] < (0.8*255)))
                        crop_N_masks = rz_N_masks[:,cr_y:cr_y+self.image_shape[0], cr_x:cr_x+self.image_shape[1]]
                        crop_N_masks_UR = (crop_N_masks[0] > (0.2*255)) & (crop_N_masks[0] < (0.8*255))
                        if (np.sum(crop_N_masks_UR) > 0.5*np_in1) or np.mean(crop_N_masks_UR) > 0.01/255.:
                            break
                else:
                    crop_N_masks = None
                    break

            if tt < 999:
                break
        crop_N_frames = rz_N_frames[:,cr_y:cr_y+self.image_shape[0], cr_x:cr_x+self.image_shape[1],:]
        
        return crop_N_frames, crop_N_masks, cr_y * (1.0 / scale), cr_x * (1.0 / scale)

    def sample_num_skip(self, sample_length, max_skip, rnd):
        skips = [0] + [rnd.randint(0, max_skip) for _ in range(sample_length-1)]
        com = [sum(skips[:i+1]) for i in range(len(skips))]
        return com

    def __getitem__(self, idx):
        info = dict()
        rnd = random.Random()
        
        _, sample_FG = self.FG[idx]
        _, sample_alpha = self.Alpha[idx]
        sample_FG_len = len(sample_FG)

        max_skip = self.max_skip
        
        ttr = 0
        while True:
            if ttr > 1000:
                return self.__getitem__(rnd.randint(0, self.__len__()-1))
            ttr += 1
            if ttr > 600:
                cum = self.sample_num_skip(self.sample_length, 0, rnd)
            else:
                cum = self.sample_num_skip(self.sample_length, max_skip, rnd)
            if (sample_FG_len-self.sample_length-cum[-1] > 1):
                break
        info['cum'] = cum

        if self.mode == 'train' and rnd.uniform(0,1) < self.do_time_flip:
            sample_FG = sample_FG[::-1]
            sample_alpha = sample_alpha[::-1]
        N_st_FG = rnd.randint(0, sample_FG_len-self.sample_length-cum[-1])
        sample_FG = [sample_FG[N_st_FG+cum_] for cum_ in cum]
        sample_alpha = [sample_alpha[N_st_FG+cum_] for cum_ in cum]

        fg, bg, a = [None] * self.sample_length, [None] * self.sample_length, [None] * self.sample_length

        # img I/O
        # FG & Alpha
        for i in range(self.sample_length):
            _f = cv2.imread(sample_FG[i], cv2.IMREAD_UNCHANGED)
            _a = cv2.imread(sample_alpha[i], cv2.IMREAD_GRAYSCALE)
            _a[_a <= 1] = 0

            # Generate FG and BG by remove color by alpha
            _bg = _f.copy()
            _af = _a.astype(np.float32) / 255.
            _af = _af[..., np.newaxis]
            _af = np.repeat(_af, 3, axis=2)
            fg_region = (_f[_af > 0].astype(np.float32) / _af[_af > 0]).astype(np.uint8)
            fg_region = np.clip(fg_region, 0, 255)
            bg_region = (_f[_af > 0].astype(np.float32) - fg_region * _af[_af >0]) / (1. - _af[_af > 0])
            np.nan_to_num(bg_region, copy=False)
            bg_region = np.clip(bg_region, 0, 255)
            bg_region = bg_region.astype(np.uint8)
            _f[_af > 0] = fg_region
            _bg[_af > 0] = bg_region
            
            fg[i] = _f
            bg[i] = _bg
            a[i] = _a
        
        # Check the results here: Oke

        if a[0].sum() < 1:
            return self.__getitem__(rnd.randint(0, self.__len__()-1))

        for i in range(self.sample_length):
            fg[i] = np.float32(fg[i])
            a[i] = np.float32(a[i])
            bg[i] = np.float32(bg[i])
            
        fg, a, scr_y, scr_x = self.random_crop(fg, a, self.sample_length, rnd)
        if bg[0] is not None:
            bg, _, scr_y, scr_x = self.random_crop(bg, None, self.sample_length, rnd)

        # gamma augmentation
        if (rnd.uniform(0,1) < self.do_gamma_aug):
            fg_aug = self.pixel_aug_gamma.to_deterministic()
            for i in range(self.sample_length):
                fg[i] = np.float32(fg_aug.augment_image(np.uint8(fg[i])))
        
        if (rnd.uniform(0,1) < self.do_gamma_aug) and (bg[0] is not None):
            bg_aug = self.pixel_aug_gamma.to_deterministic()
            for i in range(self.sample_length):
                bg[i] = np.float32(bg_aug.augment_image(np.uint8(bg[i])))

        if (rnd.uniform(0,1) < self.do_histogram_matching) and (bg[0] is not None):
            ratio = rnd.uniform(0,0.5)
            if rnd.uniform(0,1) < 0.05:
                bg_match = exposure.match_histograms(bg, fg, channel_axis=-1)
                bg = bg_match * ratio + bg * (1. - ratio)
            else:
                fg_match = exposure.match_histograms(fg, bg, channel_axis=-1)
                fg = fg_match * ratio + fg * (1. - ratio)
                    
        # random H flip 
        if rnd.randint(0,1) == 0:
            fg = _flip_axis(fg, 2)
            a = _flip_axis(a, 2)
        if rnd.randint(0,1) == 0 and (bg[0] is not None):
            bg = _flip_axis(bg, 2)


        # motion augmentation
        if (rnd.uniform(0,1) < self.do_motion_aug):
            if rnd.uniform(0,1) < 0.5 and (bg[0] is not None):
                N_cat = np.concatenate([fg, bg, a[:,:,:,np.newaxis]], axis=3) # t,h,w,7
                N_cat = N_cat.transpose((1,2,3,0)) # h,w,7,t
                N_cat = N_cat.reshape(self.image_shape[0], self.image_shape[1], -1) # h,w,7*t
                N_cat_aug = self.motion_aug(image=N_cat)["image"] # h,w,7*t
                N_cat_aug = N_cat_aug.reshape(self.image_shape[0], self.image_shape[1], -1, self.sample_length) # h,w,7,t
                N_cat_aug = N_cat_aug.transpose((3,0,1,2)) # t,h,w,7
                fg = N_cat_aug[..., :3]
                bg = N_cat_aug[..., 3:6]
                a = N_cat_aug[..., 6]
                fg = np.clip(fg, 0, 255)
                bg = np.clip(bg, 0, 255)
                a = np.clip(a, 0, 255)
            else:
                if rnd.uniform(0,1) < 0.9:
                    N_cat = np.concatenate([fg, a[:,:,:,np.newaxis]], axis=3) # t,h,w,7
                    N_cat = N_cat.transpose((1,2,3,0)) # h,w,7,t
                    N_cat = N_cat.reshape(self.image_shape[0], self.image_shape[1], -1) # h,w,7*t
                    N_cat_aug = self.motion_aug(image=N_cat)["image"] # h,w,7*t
                    N_cat_aug = N_cat_aug.reshape(self.image_shape[0], self.image_shape[1], -1, self.sample_length) # h,w,7,t
                    N_cat_aug = N_cat_aug.transpose((3,0,1,2)) # t,h,w,7
                    fg = N_cat_aug[..., :3]
                    a = N_cat_aug[..., 3]
                    fg = np.clip(fg, 0, 255)
                    a = np.clip(a, 0, 255)
                if rnd.uniform(0,1) < 0.3 and (bg[0] is not None):
                    N_cat = bg # t,h,w,7
                    N_cat = N_cat.transpose((1,2,3,0)) # h,w,7,t
                    N_cat = N_cat.reshape(self.image_shape[0], self.image_shape[1], -1) # h,w,7*t
                    N_cat_aug = self.motion_aug(image=N_cat)["image"] # h,w,7*t
                    N_cat_aug = N_cat_aug.reshape(self.image_shape[0], self.image_shape[1], -1, self.sample_length) # h,w,7,t
                    N_cat_aug = N_cat_aug.transpose((3,0,1,2)) # t,h,w,7
                    bg = N_cat_aug
                    bg = np.clip(bg, 0, 255)

        # augmentation
        if (rnd.uniform(0,1) < self.do_gaussian_aug):
            aug = self.pixel_aug_gaussian.to_deterministic()
            for i in range(self.sample_length):
                fg[i] = np.float32(aug.augment_image(np.uint8(fg[i])))
                if bg[0] is not None:
                    bg[i] = np.float32(aug.augment_image(np.uint8(bg[i])))
        if (rnd.uniform(0,1) < self.do_jpeg_aug):
            aug = self.jpeg_aug.to_deterministic()
            for i in range(self.sample_length):
                fg[i] = np.float32(aug.augment_image(np.uint8(fg[i])))
                a[i] = np.float32(aug.augment_image(np.uint8(a[i])))
                if bg[0] is not None:
                    bg[i] = np.float32(aug.augment_image(np.uint8(bg[i])))

        # random affine
        ignore_region = np.ones_like(a)
        if rnd.uniform(0,1) < self.do_affine:
            if bg[0] is not None:
                list_FM = list(fg) + list(a) + list(ignore_region) + list(bg)
            else:
                list_FM = list(fg) + list(a) + list(ignore_region)
            list_trans_FM = random_transform(list_FM, rnd, rt=10, sh=5, zm=[0.95,1.05], sc= [1, 1], cs=0.03*255., hf=False)
            fg = np.stack(list_trans_FM[:self.sample_length], axis=0)
            a = np.stack(list_trans_FM[self.sample_length:int(self.sample_length*2)], axis=0)
            ignore_region = np.stack(list_trans_FM[int(self.sample_length*2):int(self.sample_length*3)], axis=0)
            if bg[0] is not None:
                bg = np.stack(list_trans_FM[int(self.sample_length*3):int(self.sample_length*4)], axis=0)
        
        a = a / 255.

        fg = torch.from_numpy(np.transpose(fg, (0, 3, 1, 2)).copy()).float()
        if bg[0] is not None:
            bg = torch.from_numpy(np.transpose(bg, (0, 3, 1, 2)).copy()).float()
        else:
            bg = fg.clone()
        a = torch.from_numpy(a.copy()).unsqueeze(1).float()
        ignore_region = ignore_region < 0.5
        ignore_region = torch.from_numpy(ignore_region.copy()).unsqueeze(1).bool()

        max_trimap_kernel_size = 13
        eps = rnd.uniform(0.01,0.2)
        tri, a = make_trimap(rnd, a, eps=eps, dilation_kernel=rnd.randint(0,max_trimap_kernel_size), close_first=rnd.uniform(0,1)<0.05, ignore_region=ignore_region)
        
        return fg, bg, a, 0, tri, torch.tensor(idx)