# MODEL ZOO

## MaGGIe
The model weights used in our paper are available at Hugging Face Model.

| **Train dataset** | **Val dataset/Subset/Mask**   | **Checkpoint**                                                                                     | **Config**                                                                        | **Test set/Subset/Mask**  | **MAD** | **Grad** | **dtSSD** |
|-------------------|-------------------------------|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|---------------------------|---------|----------|-----------|
| I-HIM50K          | HIM2K+M-HIM2K/comp/r50_fpn_3x | [chuonghm/maggie-image-him50k-cvpr24]( https://huggingface.co/chuonghm/maggie-image-him50k-cvpr24) | [maggie_image.yaml](../configs/maggie_image.yaml) | HIM2K+M-HIM2K/natural/all | 27.17   | 9.94     | -         |
| V-HIM2K5          | V-HIM60/comp_medium/xmem      | [chuonghm/maggie-video-vim2k5-cvpr24]( https://huggingface.co/chuonghm/maggie-video-vim2k5-cvpr24) | [maggie_video.yaml](https://github.com/hmchuong/maggie/configs/maggie_video.yaml) | V-HIM60/comp_hard/xmem    | 21.23   | 7.08     | 29.90     |


## Other baselines
Coming soon...
