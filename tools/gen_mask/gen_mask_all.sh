export CUDA_VISIBLE_DEVICES=0 
sh gen_mask_single.sh ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml r50_c4_3x
sh gen_mask_single.sh ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml r50_dc5_3x
sh gen_mask_single.sh ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml r50_fpn_3x
sh gen_mask_single.sh ../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml r101_fpn_3x
sh gen_mask_single.sh ../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml r101_c4_3x
sh gen_mask_single.sh ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml x101_fpn_3x
sh gen_mask_single.sh ../configs/new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ.py r50_fpn_400e
sh gen_mask_single.sh ../configs/new_baselines/mask_rcnn_regnetx_4gf_dds_FPN_400ep_LSJ.py regnetx_400e
sh gen_mask_single.sh ../configs/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py regnety_400e
sh gen_mask_single.sh ../configs/new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py r101_fpn_400e