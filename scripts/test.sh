CUDA_VISIBLE_DEVICES=1 python -m tools.main --config /home/chuongh/vm2m/output/baseline_rn34_0614/config.yaml --eval-only \
        name baseline_rn34_0613 \
        model.weights /home/chuongh/vm2m/output/baseline_rn34_0614/best_model.pth \
        dataset.test.split test \
        test.save_results False \
        test.postprocessing True \
        test.save_dir /home/chuongh/vm2m/output/baseline_rn34_0614/vis_test