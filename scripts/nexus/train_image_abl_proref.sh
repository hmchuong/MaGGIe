#!/bin/bash
#SBATCH --job-name=vm2m
#SBATCH --output=logs/vm2m_%A.out
#SBATCH --error=logs/vm2m_%A.err
#SBATCH --time=12:00:00
#SBATCH --account=vulcan-abhinav
#SBATCH --qos=vulcan-high
#SBATCH --partition=vulcan-ampere
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --mem=128gb
#SBATCH --cpus-per-task=16

source /nfshomes/chuonghm/.zshrc
module load cuda/12.0.1
conda activate vm2m

NUM_GPUS=4
CONFIG=configs/HIM/nexus_ft_230126_abl_proref.yaml
NAME=i-him50k_final-c-128_dense-pro-ref_vulcan_240126

torchrun --standalone --nproc_per_node $NUM_GPUS tools/main_ddp.py \
                    --config $CONFIG --precision 16 name $NAME wandb.id 9dyu7c6n