for VIDEO_NAME in 7f5faedf8b 0a8c467cc3 1daf812218 2d33ad3935 2ff7f5744f 3bb4e10ed7 3a0d3a81b7 093c335ccc 3ae81609d6 6b1e04d00d 8dea22c533 a965504e88 bbe0256a75 d92532c7b2 f5bddf5598
do
    echo "Working on video $VIDEO_NAME"
    python -m tools.demo_ytvos --config configs/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml \
                                --checkpoint output/VideoMatte240K/sparsemat_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4/last_model.pth \
                                --data-dir /mnt/localssd/YTVOS --split test --video $VIDEO_NAME --output output/YTVOS/sparsemat --json ytvis_hq-test.json
done

# for VIDEO_NAME in f5bddf5598
# do
#     echo "Working on video $VIDEO_NAME"
#     python -m tools.demo_ytvos --config configs/VideoMatte240K/tcvom_vid240_s-768-512x512_b4-f8_100k_adamw_1e-4.yaml \
#                                 --checkpoint output/VideoMatte240K/tcvom_vid240_pre-hmm_s-768-512x512_b4-f8_100k_adamw_1e-4/best_model.pth \
#                                 --data-dir /mnt/localssd/YTVOS --split test --video $VIDEO_NAME --output output/YTVOS/tcvom --json ytvis_hq-test.json
# done