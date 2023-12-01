python finetune_vim.py \
--id FTPVM_VIM_only \
--which_model FTPVM \
--num_worker 12 \
--benchmark \
--lr 0.00001 -i 120000 \
--iter_switch_dataset 0 \
-b_vid_mat 4 -s_vid_mat 8 --seg_stop -1 \
--size 480 \
--tvloss_type temp_seg_allclass_weight