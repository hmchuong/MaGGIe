#!/bin/bash
#SBATCH --job-name=vm2m
#SBATCH --output=logs/vm2m_test1_%A.out
#SBATCH --error=logs/vm2m_test1_%A.err
#SBATCH --time=12:00:00
#SBATCH --account=vulcan-abhinav
#SBATCH --qos=vulcan-high
#SBATCH --partition=vulcan-ampere
#SBATCH --gres=gpu:rtxa4000:4
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=8

source /nfshomes/chuonghm/.zshrc
module load cuda/12.0.1
conda activate vm2m

python -m tools.main --dist --gpus 4 --config output/HIM/$1/config.yaml --eval-only \
                                                name $1 \
                                                model.weights output/HIM/$1/best_model.pth \
                                                dataset.test.split natural \
                                                dataset.test.downscale_mask False \
                                                dataset.test.mask_dir_name masks_matched_r101_fpn_400e \
                                                test.save_results True \
                                                test.save_dir output/HIM/$1/vis \
                                                test.postprocessing False \
                                                test.use_trimap True \
                                                test.temp_aggre False \
                                                test.log_iter 10