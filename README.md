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


## Contents
- [Install](#install)
- [MaGGIe Weights](#maggie-weights)
- [Demo](#demo)
- [Evaluation](#evaluation)
- [Training](#training)
- [Citation](#citation)

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
Please check [Demo](demo/README.md) for more information.

## Evaluation

Please check the [Model Zoo](docs/MODEL_ZOO.md) for all model weight information.
### M-HIM2K and HIM2K
The script [scripts/test_maggie_image.sh](scripts/test_maggie_image.sh) contains the full evaluation on the whole M-HIM2K. The `results.csv` in the log directory contains all the results needed. To get the number in the paper, you can run this command on 4 GPUs:
```bash
sh scripts/eval_image.sh configs/maggie_image.yaml 4  maggie
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
If you want to save the alpha mattes, please set `test.save_results True` and change the `test.save_dir`
### V-HIM60
The script [scripts/test_maggie_video.sh](scripts/test_maggie_video.sh) contains the full evaluation on the V-HIM60. This evaluation is only compatible with a single GPU. To get the number in the paper, you can run this command:
```bash
sh scripts/eval_video.sh configs/maggie_video.yaml maggie
```

If you want to evaluate on a subset (e.g, `easy`), you can run:
```bash
CONFIG=configs/maggie_video.yaml
SUBSET=easy
torchrun --standalone --nproc_per_node=1 tools/main.py --config $CONFIG --eval-only \
                    name eval_full \
                    dataset.test.split comp_$SUBSET \
                    test.save_results False \
                    test.log_iter 10
```
If you want to save the alpha mattes, please set `test.save_results True` and change the `test.save_dir`.

## Training
1. Please firstly follow [DATASET](docs/DATASET.md) to prepare the training data.

2. Download [pretrained weights](https://drive.google.com/file/d/1kNj33D7x7tR-5hXOvxO53QeCEC8ih3-A/view) of the encoder from [GCA-Matting](https://github.com/Yaoyi-Li/GCA-Matting?tab=readme-ov-file#models)

3. Training the image instance matting. 

It is recommended to use 4 A100-40GB GPUs or (any GPUs with VRAM>=24GB) for this step. 
Please check the [config](configs/maggie_image.yaml) and set `wandb` settings to your project.
```bash
NAME=<name of the experiment>
NGPUS=4
torchrun --standalone --nproc_per_node=$NGPUS tools/main.py \
                    --config configs/maggie_image.yaml \
                    --precision 16 name $NAME model.weights ''
```
If you want to resume training from the last checkpoint, you can turn on `train.resume_last` or set `train.resume` to the checkpoint folder you want to resume from. You can also set `wandb.id` to continue logging to the same experiment id.


4. Training the video instance matting

It is recommend to use 8 A100-80GB GPUs for this step. Please check the [config](configs/maggie_video.yaml) and set `wandb to your project.
```bash
NAME=<name of the experiment>
PRETRAINED=<best weight from previous step>
NGPUS=8
torchrun --standalone --nproc_per_node=$NGPUS tools/main.py \
                    --config configs/maggie_video.yaml \
                    --precision 16 name $NAME model.weights $PRETRAINED
```
If you want to resume training from the last checkpoint, you can turn on `train.resume_last` or set `train.resume` to the checkpoint folder you want to resume from. You can also set `wandb.id` to continue logging to the same experiment id.

## Citation
If you find MaGGIE useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{huynh2024maggie,
  title={MaGGIe: Masked Guided Gradual Human Instance Matting},
  author={Huynh, Chuong and Oh, Seoung Wug and and Shrivastava, Abhinav and Lee, Joon-Young},
  booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## Baselines
We also provide baselines' training and evaluation scripts at [BASELINES](docs/BASELINES.md)

## Terms of Use
TBD

## Acknowledgement
We thank Markus Woodson for his early project discussion. Our code is based on the [OTVM](https://github.com/Hongje/OTVM) and [MGM](https://github.com/yucornetto/MGMatting).
 
## Contact
If you have any question, please drop an email to chuonghm@umd.edu or create an issue on this repository.