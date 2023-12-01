# Human Masked Guided Instance Matting

Predict instance-awareness alpha matte for human with binary mask guidance

Authors: Chuong Huynh, Markus Woodson, Seoung Wug Oh, Joon-Young Lee
 
Summer Internship 2023

## Requirements
Install:
- Pytorch 2.0
- CUDA 12.0
- Pillow
- numpy
- opencv

or use the docker image: `docker-matrix-experiments-snapshot.dr-uw2.adobeitc.com/runai/clio-base-demo:0.06`

Other dependencies:
```
pip install -r requirements.txt
```

## Dataset

### Image matting
- Train and test data available at: `s3://a-chuonghm/I-HIM/`

### Video matting
- Synthesized data: `s3://a-chuonghm/V-HIM`
- Pexels data without alpha mattes: `s3://a-chuonghm/pexels-human-matting`

## Model checkpoints
### Pretrained checkpoint
Download the pretrained resnet (`s3://a-chuonghm/checkpoints/pretrain/model_best_resnet34_En_nomixup.pth`) and place in `pretrain` directory
### Image checkpoint
Config and the best checkpoint can be downloaded at `s3://a-chuonghm/checkpoints/image`. This is the checkpoint for the paper.

### Video checkpoint
Config and the best checkpoint can be downloaded at `s3://a-chuonghm/checkpoints/video`. This is the checkpoint for the paper.


## Important configs
Please look at the config files to understand the settings. Here are some important configs for the training, testing that you likely change.

- `name`: name of the experiment
- `output_dir`: directory of the logs. The final folder will be `<output_dir>/<name>`
### Dataset
- `dataset.test`: for validation/test dataset
- `dataset.train`: for train dataset
- `dataset.*.name`: `HIM` for image dataset and `MultiInstVideo` for video dataset
- `dataset.test.alpha_dir_name`: alpha groundtruth directory name.
- `dataset.test.mask_dir_name`: input mask directory name.
- `dataset.*.root_dir`: root directory of the data.
- `dataset.*.split`: split/name of the data

### Model
- `model.arch`: architecture in `vm2m/network/arch/__init__.py`
- `model.backbone`: encoder in `vm2m/network/backbone/__init__.py`
- `model.backbone_args`: additional arguments for the encoder.
- `model.decoder`: decoder in `vm2m/network/decoder/__init__.py`
- `model.decoder_args`: additional arguments for the decoder.
- `model.mgm.warmup_iter`: warm-up iterations in MGM architecture.
- `model.weights`: weights to initilize.
- `model.loss_alpha_grad_w`: weight for gradient loss.
- `model.loss_alpha_lap_w`: weight for laplacian loss.
- `model.loss_alpha_w`: weight for reconstruction loss.
- `model.loss_alpha_type`: type of reconstruction loss: l1, l2 or smooth_l1_{beta}.
- `model.loss_atten_w`: weight for attention loss.
- `model.loss_dtSSD_w`: weight for dtSSD loss.
- `model.loss_multi_inst_type`: type of multi-instance loss: l1, l2 or smooth_l1_{beta}.
- `model.loss_multi_inst_w`: weight for multi-instance loss.
- `model.loss_multi_inst_warmup`: apply multi-instance after n iterations.
- `model.reweight_os8`: using reweighting at OS8.

### Train
- `train.batch_size`: batch size.
- `train.max_iter`: maximum iterations.
- `train.num_workers`: number of cpu workers for dataloader/
- `train.optimizer`: optimizer, check `vm2m/engine/optim.py`.
- `train.resume`: resume path (directory) to the experiment.
- `train.resume_last`: `true` or `false`, resume last saved checkpoint.
- `train.scheduler`: lr scheduler, check `vm2m/engine/optim.py`.
- `train.seed`: training seed, -1 for random.
- `train.val_best_metric`: `name` of the metric to save the best checkpoint.
- `train.val_iter`: no. iterations for each validation.
- `train.val_metrics`: list of metrics for the validation, check available metrics at `vm2m/utils/metric.py`
- `train.vis_iter`: no. iterations for visualization.

### WanDB
- `wandb.entity`: entity name.
- `wandb.id`: exp id to resume, left it blank for the new exp.
- `wandb.project`: project name.
- `wandb.use`: true or false

### Test
- `test.metrics`: list of metrics for the validation, check available metrics at `vm2m/utils/metric.py`.
- `test.num_workers`: workers for both test and validation dataloaders.
- `test.save_results`: save the predictions or not.
- `test.save_dir`: directory to save the predictions

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

### Other baselines
#### 1. InstMatt:
Source code: 
Weights: 

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
You can use this repo

Retrained weights on V-HIM2K5:

Inference script:

#### 6. OTVM
You can use this repo

Retrained weights on V-HIM2K5:

Inference script: