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

python -m tools.main --dist --gpus 4 --config configs/polarized_matting/mgm_atten-dec.yaml --eval-only \
                                                name mgm_atten-dec_threshmask \
                                                model.weights output/HHM/mgm_atten-dec_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
                                                dataset.test.split test \
                                                dataset.test.use_thresh_mask True \
                                                test.save_results True \
                                                test.postprocessing False \
                                                test.log_iter 1 \
                                                test.save_dir output/polarized_matting/mgm_atten-dec_threshmask/vis_test

# Test HHM
python -m tools.main --dist --gpus 4 --config configs/HHM/mgm_hhm_short-768-512x512_bs32_50k_adamw_2e-4.yaml --eval-only \
                                                model.weights output/HHM/mgm_hhm_short-768-512x512_bs32_50k_adamw_2e-4/best_model.pth \
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