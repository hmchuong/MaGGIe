# DATASET

## Download available dataset

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
Coming soon...

Checking those scripts:
- image: `tools/synthesize_him_data.py`
- video: `tools/syn_vhm_0918.py`

List of FG/BG for train/test synthesizing: `tools/video_files`

- `tools/gen_mask`: for generate M-HIM2K with detectron2.

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