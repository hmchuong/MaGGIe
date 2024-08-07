# Baselines

We provide a list of python scripts to train and evaluate the baselines. Please also check [Model Zoo](MODEL_ZOO.md) for available checkpoints.

## InstMatt
Please use [InstMatt](https://github.com/nowsyn/InstMatt) to train and evaluate.

## SparseMat
### Image matting
To train:
```bash
NAME=<name of the experiment>
NGPUS=4
torchrun --standalone --nproc_per_node=$NGPUS tools/main.py \
                    --config configs/sparsemat_image.yaml \
                    --precision 16 name $NAME model.weights ''
```
To evaluate:
```bash
sh scripts/eval_image.sh configs/sparsemat_image.yaml 4 sparsemat
```

### Video matting
To train:
```bash
NAME=<name of the experiment>
PRETRAINED=<best weight from image matting>
NGPUS=8
torchrun --standalone --nproc_per_node=$NGPUS tools/main.py \
                    --config configs/sparsemat_video.yaml \
                    --precision 16 name $NAME model.weights $PRETRAINED
```
To evaluate:
```bash
sh scripts/eval_video.sh configs/sparsemat_video.yaml sparsemat
```

## MGM
### Image matting
We finetuned the model from the weights of [MGM in the wild](https://openaccess.thecvf.com/content/CVPR2023/papers/Park_Mask-Guided_Matting_in_the_Wild_CVPR_2023_paper.pdf), you can also initialize the model with [MGM](https://github.com/yucornetto/MGMatting/tree/main/code-base) if the pretrained weights are not available:

To train:
```bash
NAME=<name of the experiment>
NGPUS=4
torchrun --standalone --nproc_per_node=$NGPUS tools/main.py \
                    --config configs/mgm.yaml \
                    --precision 16 name $NAME model.weights ''
```
To evaluate:
```bash
sh scripts/eval_image.sh configs/mgm.yaml 4 mgm
```
### Video matting
To train:
```bash
NAME=<name of the experiment>
PRETRAINED=<best weight from image matting>
NGPUS=8
torchrun --standalone --nproc_per_node=$NGPUS tools/main.py \
                    --config configs/mgm_tcvom.yaml \
                    --precision 16 name $NAME model.weights $PRETRAINED
```
To evaluate:
```bash
sh scripts/eval_video.sh configs/mgm_tcvom.yaml mgm_tcvom
```

## MGM* (with stacked masks)
### Image matting

To train:
```bash
NAME=<name of the experiment>
NGPUS=4
torchrun --standalone --nproc_per_node=$NGPUS tools/main.py \
                    --config configs/mgm_stacked.yaml \
                    --precision 16 name $NAME model.weights ''
```
To evaluate:
```bash
sh scripts/eval_image.sh configs/mgm_stacked.yaml 4 mgm_stacked
```
### Video matting
To train:
```bash
NAME=<name of the experiment>
PRETRAINED=<best weight from image matting>
NGPUS=8
torchrun --standalone --nproc_per_node=$NGPUS tools/main.py \
                    --config configs/mgm_stacked_tcvom.yaml \
                    --precision 16 name $NAME model.weights $PRETRAINED
```
To evaluate:
```bash
sh scripts/eval_video.sh configs/mgm_stacked_tcvom.yaml mgm_stacked_tcvom
```

## FTP-VM
Please use [FTP-VM](https://github.com/csvt32745/FTP-VM) to train and evaluate.

## OTVM
Please use [OTVM](https://github.com/Hongje/OTVM) to train and evaluate.