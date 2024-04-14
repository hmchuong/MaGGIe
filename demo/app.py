import gradio as gr
from maskrcnn import predict_human_mask
from maggie_predictor import predict_image_alpha_matte

def inference(input_image, progress=gr.Progress()):
    progress(0, "Running MaskRCNN...")
    vis, masks = predict_human_mask(input_image)
    progress(0.5, "Running MaGGIe...")
    if masks.max() == 0:
        raise gr.Error("No person detected!")
    all_matte = predict_image_alpha_matte(input_image, masks)
    progress(1.0)
    return vis, all_matte
  
title="MaGGIe: Mask Guided Gradual Human Instance Matting"


# Image matting interface
description="The image human matting model that can handle multiple people in the image."
examples=["examples/man.jpg", "examples/woman.jpg", "examples/group1.jpg", "examples/group2.jpg"]
demo_image = gr.Interface(inference, inputs=gr.Image(type="pil", label="Input Image"), 
             outputs=[gr.Image(type="pil", label="Segmentation mask"), gr.Gallery(type="pil", label="Alpha Mattes")], 
             description=description, 
             examples=examples, 
             concurrency_limit=1)

description="The video human matting model that can handle multiple people in the video."
examples=["examples/man.jpg", "examples/woman.jpg", "examples/group1.jpg", "examples/group2.jpg"]
demo_video = gr.Interface(inference, inputs=gr.Image(type="pil", label="Input Image"), 
             outputs=[gr.Image(type="pil", label="Segmentation mask"), gr.Gallery(type="pil", label="Alpha Mattes")], 
             description=description,
             concurrency_limit=1)

demo = gr.TabbedInterface([demo_image, demo_video], tab_names=["image", "video"], title=title)
demo.launch(debug=True, share=False)