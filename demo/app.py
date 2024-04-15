import glob
import os
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from maskrcnn import predict_human_mask
from maggie_predictor import predict_image_alpha_matte, predict_video_alpha_matte

def inference_image(input_image, progress=gr.Progress()):
    progress(0, "Running MaskRCNN...")
    vis, masks = predict_human_mask(input_image)
    progress(0.5, "Running MaGGIe...")
    if masks.max() == 0:
        raise gr.Error("No person detected!")
    all_matte = predict_image_alpha_matte(input_image, masks)
    progress(1.0)
    return vis, all_matte
  
# Image matting interface
description="The image human matting model that can handle multiple people in the image. The model uses [Mask-RCNN R50-FPN-3X](https://huggingface.co/spaces/onnx/mask-rcnn) to extract human coarse masks and [MaGGIe](https://huggingface.co/chuonghm/maggie-image-him50k-cvpr24) to predict alpha mattes."
examples=["examples/man.jpg", "examples/woman.jpg", "examples/group1.jpg", "examples/group2.jpg"]
demo_image = gr.Interface(inference_image, inputs=gr.Image(type="pil", label="Input Image"), 
             outputs=[gr.Image(type="pil", label="Segmentation mask"), gr.Gallery(type="pil", label="Alpha Mattes")], 
             description=description, 
             examples=examples, 
             concurrency_limit=1)


def inference_video(input_video, progress=gr.Progress()):
    
    progress(0, "Running XMem...")

    # Extract binary masks with XMem
    os.system("python xmem_processor.py " + input_video)

    # Check if the masks are extracted correctly
    if not os.path.exists("video_results/mask/video0"):
        raise gr.Error("No person detected, please try video with visible human in the first frame!")

    progress(0.5, "Running MaGGIe...")
    predict_video_alpha_matte(progress, start_p=0.5, end_p=0.9)
    
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    fps = min(12, fps)

    progress(0.9, "Visualizing masks...")
    for mask_path in glob.glob("video_results/mask/video0/*/*.png"):
        mask = Image.open(mask_path).convert("L")
        mask = (np.array(mask) > 0).astype(np.uint8)
        mask = Image.fromarray(mask)
        mask.putpalette([0, 0, 0, 255, 0, 255])
        mask = mask.convert("RGB")
        image = Image.open(os.path.dirname(mask_path).replace("mask", "fgr") + ".jpg").convert("RGB")
        blended = Image.blend(image, mask, alpha=0.5)

        prefix = os.path.basename(mask_path).replace(".png", "")
        suffix = os.path.basename(os.path.dirname(mask_path))
        out_path = os.path.join("video_results/visualize_mask", f"{prefix}_{suffix}.jpg")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        blended.save(out_path)

    # Combine the mask results to video
    os.system(f"ffmpeg -framerate {fps} -pattern_type glob -i 'video_results/visualize_mask/*.jpg' -c:v libx264 -r {fps} -pix_fmt yuv420p -y video_results/xmem.mp4")

    # Combine the matte results to video
    progress(0.95, "Visualizing mattes...")
    os.system(f"ffmpeg -framerate {fps} -pattern_type glob -i 'video_results/matte/*.jpg' -c:v libx264 -r {fps} -pix_fmt yuv420p -y video_results/matte.mp4")
    
    progress(1.0)

    return "video_results/xmem.mp4", "video_results/matte.mp4"


description="The video human matting model that can handle multiple people in the video. The model runs [Mask-RCNN R50-FPN-3X](https://huggingface.co/spaces/onnx/mask-rcnn) on the first frame and [XMem](https://github.com/hkchengrex/XMem) to propagate the masks to the rest of the video. The model uses [MaGGIe](https://huggingface.co/chuonghm/maggie-video-vim2k5-cvpr24) to predict alpha mattes."
demo_video = gr.Interface(inference_video, inputs=gr.Video(format="mp4", label="Input Video", sources=["upload"]), 
             outputs=[gr.Video(format="mp4", label="Segmentation mask"), gr.Video(format="mp4", label="Alpha Mattes")], 
             description=description,
             examples=["examples/video_test1.mp4", "examples/video_test2.mp4", "examples/video_test3.mp4"],
             concurrency_limit=1)


title="MaGGIe: Mask Guided Gradual Human Instance Matting"
demo = gr.TabbedInterface([demo_image, demo_video], tab_names=["image", "video"], title=title)
demo.launch(debug=True, share=True)