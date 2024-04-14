# MODEL ZOO

## MaGGIe
The model weights used in our paper are available at Hugging Face Model.

| **Train dataset** | **Val dataset/Subset/Mask**   | **Checkpoint**                                                                                     | **Config**                                                                        | **Test set/Subset/Mask**  | **MAD**↓ | **Grad**↓ | **dtSSD**↓ |
|-------------------|-------------------------------|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|---------------------------|---------|----------|-----------|
| I-HIM50K          | HIM2K+M-HIM2K/comp/r50_fpn_3x | [chuonghm/maggie-image-him50k-cvpr24]( https://huggingface.co/chuonghm/maggie-image-him50k-cvpr24) | [maggie_image.yaml](../configs/maggie_image.yaml) | HIM2K+M-HIM2K/natural/all | 27.17   | 9.94     | -         |
| V-HIM2K5          | V-HIM60/comp_medium/xmem      | [chuonghm/maggie-video-vim2k5-cvpr24]( https://huggingface.co/chuonghm/maggie-video-vim2k5-cvpr24) | [maggie_video.yaml](../configs/maggie_video.yaml) | V-HIM60/comp_hard/xmem    | 21.23   | 7.08     | 29.90     |


## Other baselines
We provide baseline's weights in [Google Drive](https://drive.google.com/drive/folders/1r8Jgjb0w4ZfecEpkxwr7o-XVXrV3wGwu?usp=sharing). Please check [BASELINES](BASELINES.md) for training and evaluation. For quantitative results, please check the paper

Image baselines include:
- [InstMatt](https://drive.google.com/drive/folders/1_pvkrTvn40lb_fc9ruyunIqHejJguxOP?usp=drive_link)
- [SparseMat](https://drive.google.com/file/d/1GMXntzf5j4zUq53k4fnJnqSQjiUUUde6/view?usp=drive_link)
- [MGM](https://drive.google.com/file/d/1pS5zUr0HwwdToWWe3EV7jIaJyD_5Cddn/view?usp=drive_link)
- [MGM*](https://drive.google.com/file/d/15OoytUy_sJu9FfiiojhlBiTGsI3EFlRY/view?usp=drive_link)

Video matting include:
- [InstMatt](https://drive.google.com/drive/folders/1RRsHsmHYfLlH_sj3BM8LdVMMApYm0nfm?usp=drive_link)
- [SparseMat](https://drive.google.com/file/d/1hZbKiceVJd55nzuNfBmbwYcO-gdwuGOC/view?usp=drive_link)
- [MGM+TCVOM](https://drive.google.com/file/d/1fIgnWWupKZNRaWBHToK1sqac5ckTrncS/view?usp=drive_link)
- [MGM*+TCVOM](https://drive.google.com/file/d/16w24Je6n-Q776t8pG8CegHq74StcVZTV/view?usp=drive_link)
- [FTP-VM](https://drive.google.com/file/d/1l_p_1ZuQuhzkmF6OH--QsNprQzRHCKBS/view?usp=drive_link)
- [OTVM](https://drive.google.com/file/d/1UQQP9h37029f1rWwBrnDQ-UZTE4z5Lln/view?usp=drive_link)


