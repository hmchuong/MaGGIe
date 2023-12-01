import glob
from .videomatte import *

class VIMDataset(Dataset):
    def __init__(self,
                 videomatte_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 transform:MotionAugmentation=None,
                 is_VM108=True,
                 mode='train',
                 bg_num=1,
                 get_bgr_phas=False,
                 random_memtrimap=False):
        assert mode in ['train', 'val']
        self.random_memtrimap = random_memtrimap
        self.bg_num = bg_num
        self.get_bgr_phas = get_bgr_phas

        self.background_image_files = []
        self.is_bg_img = False

        self.is_VM108 = is_VM108

        self.videomatte_frames = [] # [[frame paths] x n_video]
        self.videomatte_alphas = [] # [[alpha paths] x n_video]
        self.videomatte_idx = [] # [(clip_id, frame_id)] x n_frame

        subdir = "train" if mode == 'train' else "becnhmark/comp_medium"
        
        # Load FG, pairs for each instance
        video_names = os.listdir(os.path.join(videomatte_dir, subdir, "fgr"))
        video_names.sort()
        self.videomatte_clips = video_names

        for i_video, video_name in enumerate(video_names):
            video_path = os.path.join(videomatte_dir, subdir, "fgr", video_name)
            frame_names = os.listdir(video_path)
            frame_names.sort()
            all_alpha_paths = glob.glob(os.path.join(videomatte_dir, subdir, "pha", video_name) + "/*/*.png")
            num_instances = len(all_alpha_paths) // len(frame_names)
            FG_current = [os.path.join(video_path, frame_name) for frame_name in frame_names]
            for i_instance in range(num_instances):   
                Alpha_current = [fg_path.replace("/fgr/", "/pha/").replace(".jpg", "") + f"/{i_instance:02d}.png" for fg_path in FG_current]
                self.videomatte_frames.append(FG_current)
                self.videomatte_alphas.append(Alpha_current)
                self.videomatte_idx.extend([(i_video, i_frame) for i_frame in range(len(FG_current))])

        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform

        print("VideoInstanceMatteDataset Loaded ====================")
        print("FG clips & frames: %d, %d" % (len(self.videomatte_clips), len(self.videomatte_idx)))

    def __len__(self):
        return len(self.videomatte_idx)

    def __getitem__(self, idx):
        
        fgrs, bgrs, phas = self._get_videomatte(idx)
        
        if self.transform is not None:
            ret = self.transform(fgrs, phas, bgrs)
            fgrs, phas, bgrs = ret
        
        if random.random() < 0.1:
            # random non fgr for memory frame
            fgrs[0].zero_()
            phas[0].zero_()
            
        # return fgrs, phas, bgrs
        data = {
            'fg': fgrs,
            'bg': bgrs,
            # 'rgb': fgrs*phas + bgrs*(1-phas),
            'gt': phas,
        }

        if self.random_memtrimap:
            data['trimap'] = get_dilated_trimaps(phas, 17, random_kernel=False)
            data['mem_trimap'] = get_dilated_trimaps(phas[[0]], np.random.randint(1, 16)*2+1, random_kernel=True)
        else:
            data['trimap'] = get_dilated_trimaps(phas, np.random.randint(1, 16)*2+1, random_kernel=True)
        
        return data
    
    def _get_videomatte(self, idx):
        clip_idx, frame_idx = self.videomatte_idx[idx]
        clip = self.videomatte_clips[clip_idx]
        frame_count = len(self.videomatte_frames[clip_idx])
        fgrs, bgrs, phas = [], [], []
        for i in self.seq_sampler(self.seq_length):
            frame = self.videomatte_frames[clip_idx][(frame_idx + i) % frame_count]
            alpha = self.videomatte_alphas[clip_idx][(frame_idx + i) % frame_count]
        
            _f = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
            _a = cv2.imread(alpha, cv2.IMREAD_GRAYSCALE)
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
            
            fgr = Image.fromarray(_f[:, :, ::-1])
            pha = Image.fromarray(_a)
            bgr = Image.fromarray(_bg[:, :, ::-1])


            fgr = self._downsample_if_needed(fgr)
            pha = self._downsample_if_needed(pha)
            bgr = self._downsample_if_needed(bgr)
            fgrs.append(fgr)
            bgrs.append(bgr)
            phas.append(pha)
        return fgrs, bgrs, phas
    
    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size*2:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img