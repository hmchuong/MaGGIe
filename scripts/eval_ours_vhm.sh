python -m tools.main --eval-only --config output/VHM/$1/config.yaml \
                    model.weights output/VHM/$1/last_model.pth \
                    test.save_results True \
                    test.save_dir output/VHM/$1/vis_nofusion_last/$2 \
                    dataset.test.split $2 model.decoder_args.temp_method bi