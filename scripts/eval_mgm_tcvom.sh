# CUDA_VISIBLE_DEVICES=2
# for SUBSET in comp_easy comp_medium comp_hard real
# do
#     python -m tools.main --eval-only --config output/VHM/mgm-tcvom_1024/config.yaml \
#         dataset.test.mask_dir_name xmem \
#         dataset.test.split $SUBSET \
#         model.weights output/VHM/mgm-tcvom_1024/best_model.pth \
#         test.save_results True \
#         test.save_dir output/VHM/mgm-tcvom_1024/vis/$SUBSET
# done

python -m tools.main --eval-only --config output/VHM/mgm-tcvom_1024/config.yaml \
       dataset.test.mask_dir_name xmem \
       dataset.test.alpha_dir_name xmem \
       dataset.test.split real_qual_filtered \
       model.weights output/VHM/mgm-tcvom_1024/best_model.pth \
       test.save_results True \
       test.save_dir output/VHM/mgm-tcvom_1024/vis/real_qual_filtered