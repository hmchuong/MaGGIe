CUDA_VISIBLE_DEVICES=1 python -m tools.main --config /home/chuongh/vm2m/output/baseline_rn34_0613/config.yaml --eval-only \
        name baseline_rn34_0613 \
        model.weights /home/chuongh/vm2m/output/baseline_rn34_0613/best_model.pth \
        dataset.test.split test \
        test.save_dir /home/chuongh/vm2m/output/baseline_rn34_0613/vis_test