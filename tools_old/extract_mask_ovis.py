import os
import json
import numpy as np
import collections
from PIL import Image
from pycocotools import mask as maskUtils
from tqdm import tqdm
from multiprocessing import Pool

# Process OVIS videos
data = json.load(open("/home/chuongh/vm2m/data/OVIS/annotations_train.json", "r"))
videoId2Name = {}
person_videos_ovis = set()
for video in data["videos"]:
    videoId2Name[video["id"]] = video["file_names"][0].split("/")[0]
for anno in data['annotations']:
    cate = anno['category_id']
    if cate == 1 and anno['iscrowd'] == 0:
        person_videos_ovis.add(videoId2Name[anno['video_id']])

# Save annotation of person videos

# For YTVIS 2021
out_dir = "/home/chuongh/vm2m/data/OVIS/masks"
videoId2Name = {}
videoId2FileNames = {}
for video in data["videos"]:
    videoId2Name[video["id"]] = video["file_names"][0].split("/")[0]
    videoId2FileNames[video["id"]] = video["file_names"]

video2Annos = collections.defaultdict(list)
for anno in data["annotations"]:
    video_name = videoId2Name[anno["video_id"]]
    cate = anno["category_id"]
    if cate != 1 or video_name not in person_videos_ovis or anno['iscrowd'] == 1:
        continue
    video2Annos[anno["video_id"]].append(anno)

def process(annos):
    no_inst = 0
    for anno in annos:
        width, height = anno["width"], anno["height"]
        segms = anno["segmentations"]
        frame_names = videoId2FileNames[anno["video_id"]]
        inst_id = no_inst
        no_inst += 1
        for seg, frame_name in zip(segms, frame_names):
            if seg is None:
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                mask = maskUtils.decode(seg)
            
            out_name = frame_name.replace(".jpg", "")
            out_path = os.path.join(out_dir, out_name, f"{inst_id:02d}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            mask = Image.fromarray(mask * 255)
            mask.save(out_path)

with Pool(80) as p:
    pbar = tqdm(total=len(video2Annos))
    for _ in p.imap_unordered(process, video2Annos.values()):
        pbar.update()