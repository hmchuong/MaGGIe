python -m tools.main --eval-only --config $1 \
                    model.weights $2 \
                    test.save_results False \
                    dataset.test.split $3 \
                    model.decoder_args.temp_method bi_fusion \
                    dataset.test.alpha_dir_name pha \
                    dataset.test.mask_dir_name xmem

# python -m tools.main --eval-only --config output/VHM/$1/config.yaml \
#                     model.weights output/VHM/ours_ss_1112_context-token/model_7_8.pth \
#                     test.save_results True \
#                     test.save_dir output/VHM/$1/vis_ss/$2 \
#                     dataset.test.split $2 model.decoder_args.temp_method bi_fusion \
#                     dataset.test.alpha_dir_name xmem dataset.test.mask_dir_name xmem \
#                     model.decoder_args.context_token True


# python -m tools.main --eval-only --config output/VHM/ours_vhm_bi-temp_1108_2/config.yaml \
#                     model.weights output/HIM/ours_1110_stronger-aug_guidance_scratch/last_model_24k.pth \
#                     dataset.test.mask_dir_name xmem \
#                     test.save_results True \
#                     model.arch MGM \
#                     model.decoder res_shortcut_attention_spconv_decoder_22_new \
#                     test.save_dir output/VHM/ours_vhm_bi-temp_1108_2/vis_image_model/comp_medium \
#                     dataset.test.split comp_medium                

# CUDA_VISIBLE_DEVICES=1 python -m tools.main --eval-only --config output/VHM/ours_vhm_bi-temp_1108_2/config.yaml \
#                     model.weights output/VHM/ours_ss_1114_ema_warmup_0.5_0.95_fix-masks/student_2_5.pth \
#                     test.save_results True \
#                     test.save_dir output/VHM/ours_ss_1114_ema_warmup_0.5_0.95_fix-masks/student_2_5_tokens/real_qual_filtered \
#                     dataset.test.split real_qual_filtered model.decoder_args.temp_method bi_fusion dataset.test.alpha_dir_name xmem dataset.test.mask_dir_name xmem model.decoder_args.context_token True

# python -m tools.main --eval-only --config output/VHM/ours_vhm_bi-temp_1108_2/config.yaml \
#                     model.weights output/VHM/ours_vhm_bi-temp_1108_2/best_model.pth \
#                     test.save_results True \
#                     test.save_dir output/test_debug \
#                     dataset.test.split comp_easy \
#                     model.decoder_args.temp_method bi_fusion \
#                     dataset.test.root_dir /mnt/localssd/syn/benchmark \
#                     dataset.test.alpha_dir_name pha \
#                     dataset.test.mask_dir_name xmem