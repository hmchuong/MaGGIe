CUDA_VISIBLE_DEVICES=1 python -m tools.main --config /home/chuongh/vm2m/output/baseline_rn34_0614/config.yaml --eval-only \
        name baseline_rn34_0614 \
        model.weights /home/chuongh/vm2m/output/baseline_rn34_0614/best_model.pth \
        dataset.test.split test \
        test.save_results False \
        test.postprocessing True \
        test.save_dir /home/chuongh/vm2m/output/baseline_rn34_0614/vis_test

CUDA_VISIBLE_DEVICES=0 python -m tools.main --dist --gpus 4 --config /home/chuongh/vm2m/output/baseline_rn34_0614/config.yaml --eval-only \
        name vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4 \
        model.weights output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
        dataset.test.split test \
        test.save_results False \
        test.postprocessing True \
        test.save_dir output/VideoMatte240K/vm2m_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/vis_test