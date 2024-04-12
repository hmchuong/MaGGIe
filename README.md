# <img src="figs/maggie.png" alt="maggie" width="28"/> MaGGIe: Mask Guided Gradual Human Instance Matting
[[Project Page](https://maggie-matt.github.io/)] [[Hugging Face Demo]()] [[Paper]()] [[Model Zoo](docs/MODEL_ZOO.md)] [[Datasets](docs/DATASET.md)]

*Instance-awareness alpha human matting with binary mask guidance for images and video*

**Accepted at CVPR 2024**

**[Chuong Huynh](https://hmchuong.github.io/), [Seoung Wug Oh](https://sites.google.com/view/seoungwugoh/), [Abhinav Shrivastava](https://www.cs.umd.edu/~abhinav/), [Joon-Young Lee](https://joonyoung-cv.github.io/)**

Work is a part of Summer Internship 2023 at [Adobe Research](https://research.adobe.com/)

<img src="figs/teaser.gif" alt="maggie" width="800"/>

## Release
- [2024/04/10] Demo on Huggingface is ready!
- [2024/04/07] Code, dataset and paper are released!
- [2024/04/04] Webpage is up!


## Table of Content

## Install

We tested our model on Linux CUDA 12.0, for other OS, the framework should work fine!

1. Clone this repository and navigate to `MaGGIe` folder:
```bash
git clone https://github.com/hmchuong/MaGGIe.git
cd MaGGIe
```

2. Make sure you install CUDA 12.0 and install dependencies via:

```
conda create -n maggie python=3.8 pip
conda activate maggie
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## MaGGIe Weights
Please check our [Model Zoo](docs/MODEL_ZOO.md) for all public MaGGIe checkpoints, and instructions for how to use weights.

## Demo

## Train
### 1. Please firstly follow [datasets](docs/DATASET.md) to prepare training data.
### 2. Download [pretrained weights](https://drive.google.com/file/d/1kNj33D7x7tR-5hXOvxO53QeCEC8ih3-A/view) of encoder from [GCA-Matting](https://github.com/Yaoyi-Li/GCA-Matting?tab=readme-ov-file#models)
### 3. Training the image instance matting. 

It is recommended to use 4 A100-40GB GPUs for this step. 
Please check the [config](configs/maggie_image.yaml) and set `wandb` settings to your project.
```bash
NAME=<name of the experiment>
torchrun --standalone --nproc_per_node=4 tools/main.py \
                    --config configs/maggie_image.yaml --precision 16 name $NAME
```
### 4. Training the video instance matting

It is recommend to use 8 A100-80GB GPUs for this step.
Please check the [config]()

## Evaluation

### M-HIM2K and HIM2K
The script [scripts/test_maggie_image.sh](scripts/test_maggie_image.sh) contains the full evaluation on the whole M-HIM2K. The `results.csv` in the log directory contains all the results needed. To get the number in the paper, you can run this command:
```bash
sh scripts/test_maggie_image.sh configs/maggie_image.yaml 4
```


You can also run one subset (e.g, `natural`) with one model mask (e.g, `r50_c4_3x`) by:
```bash
NGPUS=4
CONFIG=configs/maggie_image.yaml
SUBSET=natural
MODEL=r50_c4_3x
torchrun --standalone --nproc_per_node=$NGPUS tools/main.py --config $CONFIG --eval-only \
                                                name eval_full \
                                                dataset.test.split $SUBSET \
                                                dataset.test.downscale_mask False \
                                                dataset.test.mask_dir_name masks_matched_${MODEL} \
                                                test.save_results False \
                                                test.postprocessing False \
                                                test.log_iter 10
```

### V-HIM60



## Model checkpoints
### Pretrained checkpoint
Download the pretrained resnet (`s3://a-chuonghm/checkpoints/pretrain/model_best_resnet34_En_nomixup.pth`) and place in `pretrain` directory
### Image checkpoint
Config and the best checkpoint can be downloaded at `s3://a-chuonghm/checkpoints/image`. This is the checkpoint for the paper.

### Video checkpoint
Config and the best checkpoint can be downloaded at `s3://a-chuonghm/checkpoints/video`. This is the checkpoint for the paper.


## Training
The model contains two stages: training on image matting I-HIM and training on video matting V-HIM. Assuming you use RunAI with distributed training.

We recommend using 4 GPUs A100 40GB to train image and 8 GPUs A100 80GB to train video
```bash
CONFIG=<config file>
NAME=<name of the experiment>
if [ -n "$WORLD_SIZE" ] && [ "$WORLD_SIZE" -gt 1 ]; then
    PYCMD="--nproc_per_node=$RUNAI_NUM_OF_GPUS --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
else
    PYCMD="--standalone --nproc_per_node=$RUNAI_NUM_OF_GPUS"
fi

torchrun $PYCMD tools/main_ddp.py \
                    --config $CONFIG --precision 16 name $NAME
```

## Test
To test the image model on the whole M-HIM2K:
```bash
sh scripts/test_ours_image.sh <config file> <model file>
```
the script will evaluate the model 20 times, one for each subset (natural/comp and mask input)

To test the video model on V-HIM60
```bash
sh scripts/test_ours_video.sh <config file> <model file> <split>
```
where split is `comp_easy`, `comp_medium`, or `comp_hard`. The mask `xmem` will be used.

## Misc
### Synthesize data
Checking those scripts:
- image: `tools/synthesize_him_data.py`
- video: `tools/syn_vhm_0918.py`

List of FG/BG for train/test synthesizing: `tools/video_files`

### Visualize results
Those files would be helpful:
- `tools/visualize_him2k_images.py` for image visualization between methods.
- `tools/notebooks/vis_video_results.ipynb` for visualize video results.
- `tools/notebooks/process_video_website.ipynb` for processing the website videos. 
- `tools/gen_mask`: for generate M-HIM2K with detectron2.

### Other baselines
#### 1. InstMatt:
Source code: https://github.com/nowsyn/InstMatt

Updated scripts: `tools/InstMatt`

Weights: 
- Image: `s3://a-chuonghm/checkpoints/baselines/image/instmatt/`
- Video: `s3://a-chuonghm/checkpoints/baselines/video/instmatt/`

Inference script:

#### 2. SparseMat
You can use this repo

Weights and config:
- Image: `s3://a-chuonghm/checkpoints/baselines/image/sparsemat/`
- Video: `s3://a-chuonghm/checkpoints/baselines/video/sparsemat/`

#### 3. MGM
You can use this repo

Weights and config:
- Converted from MGM-In-The-Wild: `s3://a-chuonghm/checkpoints/baselines/image/mgm_wild/`
- Image: `s3://a-chuonghm/checkpoints/baselines/image/mgm/`
- Video (+TCVOM): `s3://a-chuonghm/checkpoints/baselines/video/mgm_tcvom/`

#### 4. MGM Stacked masks
You can use this repo

Weights and config:
- Image: `s3://a-chuonghm/checkpoints/baselines/image/mgm_stacked/`
- Video (+TCVOM): `s3://a-chuonghm/checkpoints/baselines/video/mgm_stacked_tcvom/`

#### 5. FTP-VM
Source code: https://github.com/csvt32745/FTP-VM

Updated scripts: `tools/FTP-VM`

Finetuned weights on V-HIM2K5: `s3://a-chuonghm/checkpoints/baselines/video/ftp-vm/`


#### 6. OTVM
Source code: https://github.com/Hongje/OTVM

Update scripts: `tools/OTVM`

Finetuned weights on V-HIM2K5: `s3://a-chuonghm/checkpoints/baselines/video/otvm/`