# CUDA_VISIBLE_DEVICES=1 python -m tools.main --config /home/chuongh/vm2m/output/baseline_rn34_0614/config.yaml --eval-only \
#         name baseline_rn34_0614 \
#         model.weights /home/chuongh/vm2m/output/baseline_rn34_0614/best_model.pth \
#         dataset.test.split test \
#         test.save_results False \
#         test.postprocessing True \
#         test.save_dir /home/chuongh/vm2m/output/baseline_rn34_0614/vis_test

# CUDA_VISIBLE_DEVICES=0 python -m tools.main --dist --gpus 4 --config /home/chuongh/vm2m/output/baseline_rn34_0614/config.yaml --eval-only \
#         name vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4 \
#         model.weights output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
#         dataset.test.split test \
#         test.save_results False \
#         test.postprocessing True \
#         test.save_dir output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/vis_test

# CUDA_VISIBLE_DEVICES=0 python -m tools.main --gpus 4 --config /home/chuongh/vm2m/output/baseline_rn34_0614/config.yaml --eval-only \
#         name vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4 \
#         model.weights output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
#         dataset.test.split test \
#         test.save_results False \
#         test.postprocessing True \
#         test.save_dir output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/vis_test test.log_iter 1

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m tools.main --dist --gpus 8 --config output/baseline_rn34_0614/config.yaml --eval-only         name baseline_rn34_0614    model.weights output/baseline_rn34_0614/best_model.pth         dataset.test.split test         test.save_results False         test.postprocessing True         test.save_dir output/baseline_rn34_0614/vis_test test.log_iter 5 test.num_workers 1

# for i in {1..1000}
# do
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python -m tools.main --dist --gpus 4 --config output/baseline_rn34_0614/config.yaml --eval-only \
#                                                 name baseline_rn34_0614 \
#                                                 model.weights output/baseline_rn34_0614/best_model.pth \
#                                                 dataset.test.split test \
#                                                 test.save_results False \
#                                                 test.postprocessing True \
#                                                 test.save_dir output/VideoMatte240K/baseline_rn34_0614/vis_test
# done


python -m tools.main --gpus 1 --config configs/VideoMatte240K/mgm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --eval-only \
                                                name mgm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4 \
                                                model.weights output/VideoMatte240K/mgm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                test.log_iter 1 \
                                                test.num_workers 10 \
                                                test.save_results False \
                                                test.postprocessing True \
                                                test.save_dir output/VideoMatte240K/mgm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/vis_test

python -m tools.main --gpus 1 --config configs/VideoMatte240K/mgm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --eval-only \
                                                name mgm_threshmask \
                                                model.weights output/VideoMatte240K/mgm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                dataset.test.use_thresh_mask True \
                                                test.save_results True \
                                                test.postprocessing True \
                                                test.log_iter 1 \
                                                test.num_workers 1 \
                                                test.save_dir output/VideoMatte240K/mgm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/vis_test_threshmask

python -m tools.main --gpus 8 --dist --config output/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4/config.yaml --eval-only \
                                                model.weights output/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                test.save_results True \
                                                test.postprocessing True \
                                                test.log_iter 1 \
                                                test.num_workers 1 \
                                                test.save_dir output/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4/vis_test

python -m tools.main --gpus 1 --config output/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4/config.yaml --eval-only \
                                                name sparsemat_last \
                                                model.weights output/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4/last_model.pth \
                                                dataset.test.split test \
                                                test.save_results False \
                                                test.postprocessing True \
                                                test.log_iter 1 \
                                                test.num_workers 10 \
                                                test.save_dir output/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4/vis_test_last

python -m tools.main --gpus 1 --config output/VideoMatte240K/tcvom_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/config.yaml --eval-only \
                                                model.weights output/VideoMatte240K/tcvom_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                test.save_results False \
                                                test.postprocessing True \
                                                test.log_iter 1 \
                                                test.num_workers 10 \
                                                test.save_dir output/VideoMatte240K/tcvom_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/vis_test


python -m tools.main --gpus 1 --config output/VideoMatte240K/tcvom_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/config.yaml --eval-only \
                                                name tcvom_threshmask \
                                                model.weights output/VideoMatte240K/tcvom_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                dataset.test.use_thresh_mask True \
                                                test.save_results True \
                                                test.postprocessing True \
                                                test.log_iter 1 \
                                                test.num_workers 1 \
                                                test.save_dir output/VideoMatte240K/tcvom_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/vis_testt_threshmask

python -m tools.main --gpus 1 --config output/baseline_rn34_0614/config.yaml --eval-only \
                                        name baseline_rn34_0614 \
                                        model.weights output/baseline_rn34_0614/best_model.pth \
                                        dataset.test.split test \
                                        dataset.test.use_thresh_mask True \
                                        test.save_results True \
                                        test.postprocessing True \
                                        test.log_iter 1 \
                                        test.num_workers 1 \
                                        test.save_dir output/VideoMatte240K/baseline_rn34_0614/vis_test_threshmask

python -m tools.main --gpus 1 --config output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/config.yaml --eval-only \
                                        model.weights output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                        dataset.test.split test \
                                        dataset.test.use_thresh_mask False \
                                        test.save_results True \
                                        test.postprocessing True \
                                        test.log_iter 1 \
                                        test.num_workers 10 \
                                        test.save_dir output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/vis_test

python -m tools.main --gpus 1 --config output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/config.yaml --eval-only \
                                        model.weights output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                        dataset.test.split test \
                                        dataset.test.use_thresh_mask True \
                                        test.save_results True \
                                        test.postprocessing True \
                                        test.log_iter 1 \
                                        test.num_workers 10 \
                                        test.save_dir output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/vis_test_threshmask

python -m tools.main --gpus 1 --config output/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4/config.yaml --eval-only \
                                                name sparsemat_last_threshmask \
                                                model.weights output/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4/last_model.pth \
                                                dataset.test.split test \
                                                dataset.test.use_thresh_mask True \
                                                test.save_results True \
                                                test.postprocessing True \
                                                test.log_iter 1 \
                                                test.num_workers 1 \
                                                test.save_dir output/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4/vis_test_last_threshmask

python -m tools.main --gpus 1 --config configs/VideoMatte240K/mgm_sftm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --eval-only \
                                                name mgm_sftm_threshmask \
                                                model.weights output/VideoMatte240K/mgm_sftm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                test.save_results True \
                                                dataset.test.use_thresh_mask True \
                                                test.postprocessing False \
                                                test.log_iter 1 \
                                                test.num_workers 10 \
                                                test.save_dir output/polarized_matting/mgm_sftm_threshmask/vis_test

python -m tools.main --gpus 1 --config configs/VideoMatte240K/mgm_sftm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml --eval-only \
                                                name mgm_sftm \
                                                model.weights output/VideoMatte240K/mgm_sftm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                test.save_results True \
                                                dataset.test.use_thresh_mask False \
                                                test.postprocessing False \
                                                test.log_iter 1 \
                                                test.num_workers 10 \
                                                test.save_dir output/polarized_matting/mgm_sftm/vis_test

# Test Polarized Matting
python -m tools.main --gpus 1 --config configs/polarized_matting/mgm.yaml --eval-only \
                                                model.weights output/VideoMatte240K/mgm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                test.log_iter 1 \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.save_dir output/polarized_matting/mgm/vis_test

python -m tools.main --gpus 1 --config configs/polarized_matting/mgm.yaml --eval-only \
                                                name mgm_threshmask \
                                                model.weights output/VideoMatte240K/mgm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                test.log_iter 1 \
                                                test.save_results True \
                                                dataset.test.use_thresh_mask True \
                                                test.postprocessing False \
                                                test.save_dir output/polarized_matting/mgm_threshmask/vis_test

python -m tools.main --gpus 1 --config configs/polarized_matting/mgm.yaml --eval-only \
                                                name mgm_wild \
                                                model.weights pretrain/wild_matting_converted.pth \
                                                dataset.test.split test \
                                                test.log_iter 1 \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.save_dir output/polarized_matting/mgm_wild/vis_test

python -m tools.main --gpus 1 --config configs/polarized_matting/mgm.yaml --eval-only \
                                                name mgm_wild_threshmask \
                                                model.weights pretrain/wild_matting_converted.pth \
                                                dataset.test.split test \
                                                test.log_iter 1 \
                                                test.save_results True \
                                                dataset.test.use_thresh_mask True \
                                                test.postprocessing False \
                                                test.save_dir output/polarized_matting/mgm_wild_threshmask/vis_test

python -m tools.main --gpus 1 --config configs/polarized_matting/sparsemat.yaml --eval-only \
                                                model.weights output/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4/last_model.pth \
                                                dataset.test.split test \
                                                test.log_iter 1 \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.save_dir output/polarized_matting/sparsemat/vis_test

python -m tools.main --gpus 1 --config configs/polarized_matting/sparsemat.yaml --eval-only \
                                                name sparsemat_threshmask \
                                                model.weights output/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4/last_model.pth \
                                                dataset.test.split test \
                                                dataset.test.use_thresh_mask True \
                                                test.log_iter 1 \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.save_dir output/polarized_matting/sparsemat_threshmask/vis_test

python -m tools.main --gpus 1 --config configs/polarized_matting/vm2m.yaml --eval-only \
                                        model.weights output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                        dataset.test.split test \
                                        dataset.test.use_thresh_mask False \
                                        test.save_results True \
                                        test.postprocessing True \
                                        test.log_iter 1 \
                                        test.save_dir output/polarized_matting/vm2m/vis_test

python -m tools.main --gpus 1 --config configs/polarized_matting/vm2m.yaml --eval-only \
                                        name vm2m_threshmask \
                                        model.weights output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                        dataset.test.split test \
                                        dataset.test.use_thresh_mask True \
                                        test.save_results True \
                                        test.postprocessing True \
                                        test.log_iter 1 \
                                        test.save_dir output/polarized_matting/vm2m_threshmask/vis_test

python -m tools.main --gpus 1 --config configs/polarized_matting/tcvom.yaml --eval-only \
                                                model.weights output/VideoMatte240K/tcvom_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                test.save_results False \
                                                test.postprocessing True \
                                                test.log_iter 1 \
                                                test.num_workers 10 \
                                                test.save_dir output/polarized_matting/tcvom/vis_test

python -m tools.main --gpus 1 --config configs/polarized_matting/tcvom.yaml --eval-only \
                                                name tcvom_0714 \
                                                model.weights output/VideoMatte240K/tcvom_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4_fix-dtssdloss/best_model.pth \
                                                dataset.test.split test \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.log_iter 1 \
                                                test.num_workers 10 \
                                                test.save_dir output/polarized_matting/tcvom_0714/vis_test

python -m tools.main --gpus 1 --config configs/polarized_matting/tcvom.yaml --eval-only \
                                                name tcvom_0714_threshmask \
                                                model.weights output/VideoMatte240K/tcvom_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4_fix-dtssdloss/best_model.pth \
                                                dataset.test.split test \
                                                dataset.test.use_thresh_mask True \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.log_iter 1 \
                                                test.num_workers 10 \
                                                test.save_dir output/polarized_matting/tcvom_0714_threshmask/vis_test

python -m tools.main --gpus 1 --config configs/polarized_matting/vm2m0711.yaml --eval-only \
                                                name vm2m0711 \
                                                model.weights output/VideoMatte240K/vm2m0711_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4_continue/best_model.pth \
                                                dataset.test.split test \
                                                test.save_results True \
                                                test.postprocessing True \
                                                test.log_iter 1 \
                                                test.num_workers 10 \
                                                test.save_dir output/polarized_matting/vm2m0711/vis_test

python -m tools.main --gpus 1 --config configs/polarized_matting/vm2m0711.yaml --eval-only \
                                                name vm2m0711_threshmask \
                                                model.weights output/VideoMatte240K/vm2m0711_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4_continue/best_model.pth \
                                                dataset.test.split test \
                                                test.save_results True \
                                                dataset.test.use_thresh_mask True \
                                                test.postprocessing True \
                                                test.log_iter 1 \
                                                test.num_workers 10 \
                                                test.save_dir output/polarized_matting/vm2m0711_threshmask/vis_test_postprocess


python -m tools.main --gpus 1 --config configs/polarized_matting/mgm_sftm.yaml --eval-only \
                                                name mgm_sftm \
                                                model.weights output/VideoMatte240K/mgm_sftm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                dataset.test.use_thresh_mask False \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.log_iter 1 \
                                                test.save_dir output/polarized_matting/mgm_sftm/vis_test

python -m tools.main --gpus 1 --config configs/polarized_matting/mgm_sftm.yaml --eval-only \
                                                name mgm_sftm_threshmask \
                                                model.weights output/VideoMatte240K/mgm_sftm_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                dataset.test.use_thresh_mask True \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.log_iter 1 \
                                                test.save_dir output/polarized_matting/mgm_sftm_threshmask/vis_test

python -m tools.main --gpus 4 --config configs/polarized_matting/mgm_atten-dec.yaml --eval-only \
                                                name mgm_atten-dec_threshmask \
                                                model.weights output/HHM/mgm_atten-dec_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                dataset.test.use_thresh_mask True \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.log_iter 1 \
                                                test.save_dir output/polarized_matting/mgm_atten-dec_threshmask/vis_test

python -m tools.main --gpus 4 --config configs/polarized_matting/mgm_atten-dec.yaml --eval-only \
                                                name mgm_atten-dec_threshmask \
                                                model.weights output/HHM/mgm_atten-dec_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                dataset.test.use_thresh_mask True \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.log_iter 1 \
                                                test.save_dir output/polarized_matting/mgm_atten-dec_threshmask/vis_test

python -m tools.main --gpus 4 --config configs/polarized_matting/mgm_atten-dec.yaml --eval-only \
                                                name mgm_atten-dec \
                                                model.weights output/HHM/mgm_atten-dec_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                dataset.test.use_thresh_mask False \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.log_iter 1 \
                                                test.save_dir output/polarized_matting/mgm_atten-dec/vis_test

python -m tools.main --gpus 4 --config configs/polarized_matting/mgm_m-1_atten-dec.yaml --eval-only \
                                                name mgm_m-1_atten-dec_threshmask \
                                                model.weights output/VideoMatte240K/mgm_m-1_atten-dec_vid240_pre-hmm_s-768-512x512_b6-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                dataset.test.use_thresh_mask True \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.log_iter 1 \
                                                test.save_dir output/polarized_matting/mgm_m-1_atten-dec_threshmask/vis_test

python -m tools.main --gpus 4 --config configs/polarized_matting/mgm_m-1_atten-dec.yaml --eval-only \
                                                name mgm_m-1_atten-dec \
                                                model.weights output/VideoMatte240K/mgm_m-1_atten-dec_vid240_pre-hmm_s-768-512x512_b6-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                dataset.test.use_thresh_mask False \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.log_iter 1 \
                                                test.save_dir output/polarized_matting/mgm_m-1_atten-dec/vis_test

# Test HHM
python -m tools.main --dist --gpus 4 --config configs/HHM/mgm_hhm_short-768-512x512_bs32_50k_adamw_2e-4.yaml --eval-only \
                                                model.weights output/HHM/mgm_hhm_short-768-512x512_bs32_50k_adamw_2e-4/best_model.pth \
                                                dataset.test.split val \
                                                test.log_iter 1 \
                                                test.save_results False \
                                                test.postprocessing True

python -m tools.main --dist --gpus 4 --config configs/HHM/mgm_hhm_short-768-512x512_bs32_50k_adamw_2e-4.yaml --eval-only \
                                                name mgm_wild \
                                                model.weights pretrain/wild_matting_converted.pth \
                                                dataset.test.split val \
                                                test.log_iter 1 \
                                                test.save_results False \
                                                test.postprocessing True

python -m tools.main --gpus 1 --config configs/HHM/mgmdk_8x8_hhm_short-768-512x512_bs32_50k_adamw_2e-4.yaml --eval-only \
                                                model.weights output/HHM/mgmdk_8x8_hhm_short-768-512x512_bs32_50k_adamw_2e-4/best_model.pth \
                                                dataset.test.split val \
                                                test.log_iter 1 \
                                                test.save_results False \
                                                test.postprocessing True

python -m tools.main --gpus 1 --config configs/HHM/mgm_atten-dec_hhm_short-768-512x512_bs32_50k_adamw_2e-4.yaml --eval-only \
                                                model.weights output/HHM/mgm_atten-dec_hhm_short-768-512x512_bs32_50k_adamw_2e-4/best_model.pth \
                                                dataset.test.split val \
                                                test.log_iter 1 \
                                                test.save_results False \
                                                test.postprocessing True

python -m tools.main --dist --gpus 4 --config configs/HHM/mgm_swint_atten-dec_q-16_hhm_short-768-512x512_bs30_50k_adamw_2e-4.yaml --eval-only \
                                                model.weights output/HHM/mgm_swint_atten-dec_q-16_hhm_short-768-512x512_bs20_50k_adamw_2e-4/best_model.pth \
                                                dataset.test.split val \
                                                test.log_iter 1 \
                                                test.save_results False \
                                                test.postprocessing True

# Test AIM-500
python -m tools.main --dist --gpus 4 --config configs/AIM-500/mgm.yaml --eval-only \
                                                model.weights output/HHM/mgm_hhm_short-768-512x512_bs32_50k_adamw_2e-4/best_model.pth

python -m tools.main --dist --gpus 4 --config configs/AIM-500/mgmdk_8x8.yaml --eval-only \
                                                model.weights output/HHM/mgmdk_8x8_hhm_short-768-512x512_bs32_50k_adamw_2e-4/best_model.pth


python -m tools.main --dist --gpus 4 --config configs/AIM-500/mgm_multiinst.yaml --eval-only \
                                                model.weights output/HIM/tuning_mgm_filtered/best_model.pth

python -m tools.main --dist --gpus 4 --config configs/AIM-500/mgm_atten_embed.yaml --eval-only \
                                                name mgm_atten_embed \
                                                model.weights output/HIM/tuning_enc-dec-os8-conv_filtered_multi-inst/best_model.pth

python -m tools.main --dist --gpus 4 --config configs/AIM-500/mgm_atten-dec.yaml --eval-only \
                                                model.weights output/HHM/mgm_atten-dec_hhm_short-768-512x512_bs32_50k_adamw_2e-4/best_model.pth

# Test AM-200
python -m tools.main --dist --gpus 4 --config configs/AM-2K/mgm.yaml --eval-only \
                                                model.weights output/HHM/mgm_hhm_short-768-512x512_bs32_50k_adamw_2e-4/best_model.pth

python -m tools.main --dist --gpus 4 --config configs/AM-2K/mgm_multiinst.yaml --eval-only \
                                                name mgm_multi-inst \
                                                model.weights output/HIM/tuning_mgm_filtered/best_model.pth

python -m tools.main --dist --gpus 4 --config configs/AM-2K/mgm_atten_embed.yaml --eval-only \
                                                name mgm_atten_embed \
                                                model.weights output/HIM/tuning_enc-dec-os8-conv_filtered_multi-inst/best_model.pth

python -m tools.main --dist --gpus 4 --config configs/AM-2K/mgmdk_8x8.yaml --eval-only \
                                                model.weights output/HHM/mgmdk_8x8_hhm_short-768-512x512_bs32_50k_adamw_2e-4/best_model.pth

python -m tools.main --dist --gpus 4 --config configs/AM-2K/mgm_atten-dec.yaml --eval-only \
                                                model.weights output/HHM/mgm_atten-dec_hhm_short-768-512x512_bs32_50k_adamw_2e-4/best_model.pth

# Test HIM
python -m tools.main --dist --gpus 4 --config configs/HIM/mgm_him_short-768-512x512_bs8_50k_adamw_2e-4.yaml --eval-only \
                                                name mgm_him_short-768-512x512_bs32_50k_adamw_2e-4 \
                                                model.weights output/HIM/mgm_him_max-3-inst_bs12_30k_adamw_2.0e-4/best_model.pth \
                                                dataset.test.split natural \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.save_dir output/HIM/mgm_him_max-3-inst_bs12_30k_adamw_2.0e-4/vis_natural

python -m tools.main --dist --gpus 4 --config configs/HIM/mgm_enc-embed_dec-embed-atten_him_short-768-512x512_bs8_50k_adamw_2e-4.yaml --eval-only \
                                                name mgm_him_enc-dec-embed_no-inst-loss_max-3-inst_bs12_30k_adamw_2.0e-4 \
                                                model.weights output/HIM/mgm_him_enc-dec-embed_no-inst-loss_max-3-inst_bs12_30k_adamw_2.0e-4/best_model.pth \
                                                dataset.test.split natural \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.save_dir output/HIM/mgm_him_enc-dec-embed_no-inst-loss_max-3-inst_bs12_30k_adamw_2.0e-4/vis_natural

python -m tools.main --dist --gpus 4 --config configs/HIM/mgm_him_short-768-512x512_bs8_50k_adamw_2e-4.yaml --eval-only \
                                                name mtuning_mgm_filtered \
                                                model.weights output/HIM/tuning_mgm_filtered/best_model.pth \
                                                dataset.test.split natural \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.use_trimap False \
                                                test.save_dir output/HIM/tuning_mgm_filtered/vis_natural_best_final

python -m tools.main --gpus 1 --config configs/HIM/mgm_enc-embed_dec-id-embed_him_short-768-512x512_bs8_50k_adamw_2e-4.yaml --eval-only \
                                                name tuning_enc-dec-id-embed_filtered \
                                                model.weights output/HIM/tuning_enc-dec-id-embed_filtered_multi-inst/best_model.pth \
                                                dataset.test.split natural \
                                                test.save_results True \
                                                test.postprocessing True \
                                                test.use_trimap False \
                                                test.save_dir output/HIM/tuning_enc-dec-id-embed_filtered/vis_natural_multi-inst_best_new1

python -m tools.main --dist --gpus 4 --config configs/HIM/mgm_enc-embed_dec-embed-atten_him_short-768-512x512_bs8_50k_adamw_2e-4.yaml --eval-only \
                                                name tuning_enc-dec-os8-conv_filtered_multi-inst \
                                                model.weights output/HIM/tuning_enc-dec-os8-conv_filtered_multi-inst/best_model.pth \
                                                dataset.test.split natural \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.use_trimap False \
                                                test.save_dir output/HIM/tuning_enc-dec-os8-conv_filtered_multi-inst/vis_natural_7k5

CUDA_VISIBLE_DEVICES=2,3 python -m tools.main --dist --gpus 2 --config configs/HIM/mgm_enc-embed_dec-embed-atten_him_short-768-512x512_bs8_50k_adamw_2e-4.yaml --eval-only \
                                                name tuning_enc-dec_filtered \
                                                model.weights output/HIM/tuning_enc-dec_filtered/best_model.pth \
                                                dataset.test.split natural \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.save_dir output/HIM/tuning_enc-dec_filtered/vis_natural