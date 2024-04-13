# DATASET

## Download available datasets

The dataset is available on [Hugging Face Datasets](https://huggingface.co/datasets/chuonghm/MaGGIe-HIM). Please download the set you need:

| **Dataset**     | **Image/video** | **Train/val** | **Filenames**         |
|-----------------|-------------|-----------|-----------------------|
| I-HIM50K        | Image       | Train     | I-HIM50K.z* (6 files) |
| M-HIM2K + HIM2K | Image       | Val       | HIM2K_M-HIM2K.zip     |
| V-HIM2K5        | Video       | Train     | V-HIM2K5.z* (7 files) |
| V-HIM60         | Video       | Val       | V-HIM60.zip           |

For the zip splits, you can run following command to combine before unzip:
```
# e.g, if you want to combine I-HIM50K files
zip -F I-HIM50K.zip --out I-HIM50K_all.zip

# then unzip the file
unzip I-HIM50K_all.zip
```

## How to synthesize dataset?
### I-HIM50K
You need to download the [HHM dataset](https://github.com/nowsyn/sparsemat), [BG-20K dataset](https://github.com/JizhiziLi/GFM) and [invalid_him.txt](https://huggingface.co/datasets/chuonghm/MaGGIe-HIM/blob/main/invalid_him.txt) before running this script:
```bash
python tools/synthesize_image_him.py --image-root HHM/train/images \
                                     --invalid-names invalid_him.txt \
                                     --bg-root BG-20K/train \
                                     --output-dir data/I-HIM50K \
                                     --n-workers 80
```
Here, we run the process in 80 threads, please adjust it based on your machine.

### M-HIM2K
To gen all binary masks for [HIM2K](https://github.com/nowsyn/InstMatt), we use [detectron2](https://github.com/facebookresearch/detectron2). We provide scripts to process masks in [tools/gen_mask](../tools/gen_mask). The process of generate masks are:
1. Clone the detectron2 repository.
2. Copy all files in `tools/gen_mask` to `demo` directory.
3. Download all pretrained models from [Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-instance-segmentation-baselines-with-mask-r-cnn) to `detectron2/pretrained`
4. Update data paths in `get_mask_single.sh`
5. Generate masks with `gen_mask_all.sh`

### V-HIM
You need to download those datasets
1. [VideoMatte240K](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets)
2. [VM108](https://github.com/yunkezhang/TCVOM#videomatting108-dataset)
3. [CRGNN](https://github.com/TiantianWang/VideoMatting-CRGNN)

and foreground/background list:
1. [fg_train.txt](https://huggingface.co/datasets/chuonghm/MaGGIe-HIM/blob/main/fg_train.txt)
2. [fg_test.txt](https://huggingface.co/datasets/chuonghm/MaGGIe-HIM/blob/main/fg_test.txt)
3. [bg_train.txt](https://huggingface.co/datasets/chuonghm/MaGGIe-HIM/blob/main/bg_train.txt)
4. [bg_test.txt](https://huggingface.co/datasets/chuonghm/MaGGIe-HIM/blob/main/bg_test.txt)

After organizing the data following paths in `*.txt` files, you can generate the V-HIM2K5 by running this command:
```bash
python tools/synthesize_video_him.py --split train --data-dir /path/to/video --out-dir data/V-HIM2K5 --n-workers 80
# or
python tools/synthesize_video_him.py --split test --data-dir /path/to/video --out-dir data/V-HIM60 --n-workers 16
```

To generate masks, we combine the ouput of [GenVIS](https://github.com/miranheo/GenVIS) and [XMem](https://github.com/hkchengrex/XMem). The complete code will be released in the future!

## How to use your own dataset?

You should prepare your data following these structures

### For image matting training dataset
```
<root_dir>
└── <split>
    ├── images
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── <alpha_dir_name>
    │   ├── image1
    │   │   ├── 00.png # instance matte 1
    │   │   └── 01.png # instance matte 2
    │   └── image2
    └──<mask_dir_name (optional)>
        ├── image1
        │   ├── 00.png # instance mask 1
        │   └── 01.png # instance mask 2
        └── image2
  
```
### For image matting validation/testing dataset
```
<root_dir>
├── images
|   └── <split>
│       ├── image1.jpg
│       └── image2.jpg
├── <alpha_dir_name>
|   └── <split>
│       ├── image1
│       │   ├── 00.png # instance matte 1
│       │   └── 01.png # instance matte 2
│       └── image2
└──<mask_dir_name>
    └── <split>
        ├── image1
        │   ├── 00.png # instance mask 1
        │   └── 01.png # instance mask 2
        └── image2
  
```
### For video matting dataset
```
<root_dir>
└── <split>
    ├── fgr
    │   └── video1
    │       ├── image1.jpg
    │       └── image2.jpg
    ├── <alpha_dir_name>
    │   └── video1
    │       ├── image1
    │       │   ├── 00.png # instance matte 1
    │       │   └── 01.png # instance matte 2
    │       └── image2
    └──<mask_dir_name (optional in training)>
        └── video1
            ├── image1
            │   ├── 00.png # instance mask 1
            │   └── 01.png # instance mask 2
            └── image2
```