import cv2
import os
import numpy as np
import insightface
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from multiprocessing import Pool

def check_video(video_path):
    app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider']) # enable detection model only
    app.prepare(ctx_id=0, det_size=(640, 640))
    all_frames = list(os.listdir(video_path))
    all_frames.sort()
    for frame in all_frames[::3]:
        if frame.endswith('.jpg'):
            frame_path = os.path.join(video_path, frame)
            print("processing frame:", frame_path)
            faces = app.get(cv2.imread(frame_path))
            if len(faces) > 1:
                return os.path.basename(video_path), False
    return os.path.basename(video_path), True

if __name__ == "__main__":
    invalid_videos = []
    valid_videos = []

    VIDEO_PATH = '/mnt/localssd/VideoMatte240K/train/fgr'
    all_videos = []
    for video in os.listdir(VIDEO_PATH):
        all_videos.append(os.path.join(VIDEO_PATH, video))
    with Pool(8) as p:
        for video_name, is_valid in p.imap_unordered(check_video, all_videos):
            if is_valid:
                valid_videos.append(video_name)
            else:
                invalid_videos.append(video_name)
    
    with open('valid_train_videos.txt', 'w') as f:
        for video in valid_videos:
            f.write(video + '\n')
    with open('invalid_train_videos.txt', 'w') as f:
        for video in invalid_videos:
            f.write(video + '\n')

    

