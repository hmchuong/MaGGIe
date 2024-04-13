# Baselines

We provide a list of python scripts to train and evaluate the baselines. Please also check [Model Zoo](MODEL_ZOO.md) for available checkpoints.

## InstMatt
Please use [InstMatt](https://github.com/nowsyn/InstMatt) to train and evaluate.

## SparseMat
### Image matting
To train:
```bash

```
To evaluate:
```bash

```

### Video matting
To train:
```bash
```
To evaluate:
```bash
```

## MGM
### Image matting
We finetuned the model from the weights of [MGM in the wild](https://openaccess.thecvf.com/content/CVPR2023/papers/Park_Mask-Guided_Matting_in_the_Wild_CVPR_2023_paper.pdf), you can also initialize the model with [MGM](https://github.com/yucornetto/MGMatting/tree/main/code-base) if the pretrained weights are not available:

To train:
```bash
```
To evaluate:
```bash
```
### Video matting
To train:
```bash
```
To evaluate:
```bash
```

## MGM with stacked masks
### Image matting

To train:
```bash
```
To evaluate:
```bash
```
### Video matting
To train:
```bash
```
To evaluate:
```bash
```

## FTP-VM
Please use [FTP-VM](https://github.com/csvt32745/FTP-VM) to train and evaluate.

## OTVM
Please use [OTVM](https://github.com/Hongje/OTVM) to train and evaluate.