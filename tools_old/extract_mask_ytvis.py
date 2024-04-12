import os
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as maskUtils

ytvis2019_image_dir = "/home/chuongh/vm2m/data/YTVIS/2019/train/JPEGImages"
ytvis2021_image_dir = "/home/chuongh/vm2m/data/YTVIS/2021/train/JPEGImages"

person_videos_2019 = set()

# Process 2019 videos
data = json.load(open("/home/chuongh/vm2m/data/YTVIS/2019/train.json", "r"))
videoId2Name = {}
for video in data["videos"]:
    videoId2Name[video["id"]] = video["file_names"][0].split("/")[0]

for anno in data['annotations']:
    cate = anno['category_id']
    if cate == 1:
        person_videos_2019.add(videoId2Name[anno['video_id']])

person_videos_2021 = set()
data = json.load(open("/home/chuongh/vm2m/data/YTVIS/2021/train/instances.json", "r"))
videoId2Name = {}
for video in data["videos"]:
    videoId2Name[video["id"]] = video["file_names"][0].split("/")[0]
for anno in data['annotations']:
    cate = anno['category_id']
    if cate == 26:
        person_videos_2021.add(videoId2Name[anno['video_id']])

# Save annotation of person videos

# For YTVIS 2021
out_dir = "/home/chuongh/vm2m/data/YTVIS/2021/train/masks"
data = json.load(open("/home/chuongh/vm2m/data/YTVIS/2021/train/instances.json", "r"))
videoId2Name = {}
videoId2FileNames = {}
for video in data["videos"]:
    videoId2Name[video["id"]] = video["file_names"][0].split("/")[0]
    videoId2FileNames[video["id"]] = video["file_names"]
videoId2NoInst = {}
for anno in tqdm(data["annotations"]):
    video_name = videoId2Name[anno["video_id"]]
    cate = anno["category_id"]
    if cate != 26 or video_name not in person_videos_2021:
        continue
    width, height = anno["width"], anno["height"]
    segms = anno["segmentations"]
    frame_names = videoId2FileNames[anno["video_id"]]
    videoId2NoInst[anno["video_id"]] = videoId2NoInst.get(anno["video_id"], 0) + 1
    inst_id = videoId2NoInst[anno["video_id"]] - 1
    for seg, frame_name in zip(segms, frame_names):
        if seg is None:
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask = maskUtils.decode(maskUtils.frPyObjects(seg, height, width))
        
        out_name = frame_name.replace(".jpg", "")
        out_path = os.path.join(out_dir, out_name, f"{inst_id:02d}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mask = Image.fromarray(mask * 255)
        mask.save(out_path)

print("Processing 2019 videos")
# For YTVIS 2019
out_dir = "/home/chuongh/vm2m/data/YTVIS/2019/train/masks"
data = json.load(open("/home/chuongh/vm2m/data/YTVIS/2019/train.json", "r"))
videoId2Name = {}
videoId2FileNames = {}
for video in data["videos"]:
    videoId2Name[video["id"]] = video["file_names"][0].split("/")[0]
    videoId2FileNames[video["id"]] = video["file_names"]
videoId2NoInst = {}
person_videos = person_videos_2019.difference(person_videos_2021)
for anno in tqdm(data["annotations"]):
    video_name = videoId2Name[anno["video_id"]]
    cate = anno["category_id"]
    if cate != 26 or video_name not in person_videos_2021:
        continue
    width, height = anno["width"], anno["height"]
    segms = anno["segmentations"]
    frame_names = videoId2FileNames[anno["video_id"]]
    videoId2NoInst[anno["video_id"]] = videoId2NoInst.get(anno["video_id"], 0) + 1
    inst_id = videoId2NoInst[anno["video_id"]] - 1
    for seg, frame_name in zip(segms, frame_names):
        if seg is None:
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask = maskUtils.decode(maskUtils.frPyObjects(seg, height, width))
        
        out_name = frame_name.replace(".jpg", "")
        out_path = os.path.join(out_dir, out_name, f"{inst_id:02d}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mask = Image.fromarray(mask * 255)
        mask.save(out_path)