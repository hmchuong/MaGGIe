import gradio as gr
from maskrcnn import predict_human_mask

def inference(input_image):
    vis, masks = predict_human_mask(input_image)
    return vis
  
title="MaGGIe: Mask Guided Gradual Human Instance Matting"
description="The image-video human matting model that can handle multiple people in the image."
examples=["examples/man.jpg", "examples/woman.jpg", "examples/group1.jpg", "examples/group2.jpg"]
gr.Interface(inference, inputs=gr.Image(type="pil", label="Input Image"), 
             outputs=gr.Image(type="pil", label="Segmentation mask"), 
             title=title, 
             description=description, 
             examples=examples, 
             concurrency_limit=1).launch(debug=True, share=True)