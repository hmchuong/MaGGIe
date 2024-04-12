#!/bin/bash
#SBATCH --job-name=vm2m
#SBATCH --output=logs/vm2m_test4_%A.out
#SBATCH --error=logs/vm2m_test4_%A.err
#SBATCH --time=12:00:00
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa4000:4
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=8

# for SUBSET in natural comp
# do
#     for MODEL in r50_c4_3x r50_dc5_3x r50_fpn_3x r50_fpn_400e r101_c4_3x r101_fpn_3x r101_fpn_400e regnetx_400e regnety_400e x101_fpn_3x
#     do
#     python -m tools.main --dist --gpus 4 --dist-url tcp://127.0.0.1:1234 --config $1 --eval-only \
#                                                     name m-him2k \
#                                                     model.weights $2 \
#                                                     dataset.test.split $SUBSET \
#                                                     dataset.test.downscale_mask False \
#                                                     dataset.test.mask_dir_name masks_matched_${MODEL} \
#                                                     test.save_results False \
#                                                     test.postprocessing False \
#                                                     test.use_trimap True \
#                                                     test.temp_aggre False \
#                                                     test.log_iter 10
#     done
# done

source /nfshomes/chuonghm/.zshrc
module load cuda/12.0.1
conda activate vm2m

for SUBSET in natural comp
do
    for MODEL in r50_c4_3x r50_dc5_3x r50_fpn_3x
    do
    python -m tools.main --dist --gpus 4 --config output/HIM/$1/config.yaml --eval-only \
                                                    name ${1}_eval1 \
                                                    model.weights output/HIM/$1/best_model.pth \
                                                    dataset.test.split $SUBSET \
                                                    dataset.test.downscale_mask False \
                                                    dataset.test.mask_dir_name masks_matched_${MODEL} \
                                                    test.save_results False \
                                                    test.postprocessing False \
                                                    test.use_trimap True \
                                                    test.temp_aggre False \
                                                    test.log_iter 10
    done
done