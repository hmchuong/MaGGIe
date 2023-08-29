import sys
import glob
from PIL import Image

video_dirs = []
with open('../data/VHM/train.txt', 'r') as f:
    for line in f:
        if 'VideoMatte240K/' in line:
            video_dirs.append(line.strip())

for i, video_dir in enumerate(video_dirs):
    all_names = list(glob.glob('../data/VHM/' + video_dir + '/*.jpg'))
    all_names = sorted(all_names)
    Image.open(all_names[10]).save("test.png")
    next = input(f"{i}/ {len(video_dirs)}: {video_dir}")
    if next == 'q':
        sys.exit(0)