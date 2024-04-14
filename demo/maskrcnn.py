import os
import numpy as np
import cv2
from PIL import Image
import onnxruntime as rt

def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image



# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# onnxruntime.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
if os.path.exists("MaskRCNN-10.onnx") == False:
    os.system("wget https://github.com/AK391/models/raw/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx")
sess = rt.InferenceSession("MaskRCNN-10.onnx", providers=['CUDAExecutionProvider'])

outputs = sess.get_outputs()


def display_human_segmentation(image, boxes, labels, scores, masks, score_threshold=0.7):
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio

    image = np.array(image)
    all_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    instance_id = 0
    for mask, box, label, score in zip(masks, boxes, labels, scores):
        if label != 1: continue
        # Showing boxes with score > 0.7
        if score <= score_threshold:
            continue

        # Finding contour based on mask
        mask = mask[0, :, :, None]
        int_box = [int(i) for i in box]
        mask = cv2.resize(mask, (int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1))
        mask = mask > 0.5
        im_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        x_0 = max(int_box[0], 0)
        x_1 = min(int_box[2] + 1, image.shape[1])
        y_0 = max(int_box[1], 0)
        y_1 = min(int_box[3] + 1, image.shape[0])
        mask_y_0 = max(y_0 - box[1], 0)
        mask_y_1 = mask_y_0 + y_1 - y_0
        mask_x_0 = max(x_0 - box[0], 0)
        mask_x_1 = mask_x_0 + x_1 - x_0
        im_mask[y_0:y_1, x_0:x_1] = mask[
            mask_y_0 : mask_y_1, mask_x_0 : mask_x_1
        ]
        instance_id += 1
        im_mask = np.ascontiguousarray(im_mask) * instance_id
        all_masks = np.maximum(all_masks, im_mask)
    
    pred = Image.fromarray(all_masks)
    pred.putpalette(get_palette(instance_id + 1))
    print(instance_id)
    # blend the image with the mask
    image = Image.fromarray(image)
    pred = pred.convert("RGB")
    image = image.convert("RGB")
    blended = Image.blend(image, pred, alpha=0.5)
    return blended, all_masks

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def predict_human_mask(input_image):
    input_tensor = preprocess(input_image)
  

    # Mask R-CNN
    output_names = list(map(lambda output: output.name, outputs))
    input_name = sess.get_inputs()[0].name
    
    boxes, labels, scores, masks = sess.run(output_names, {input_name: input_tensor})
    vis, masks = display_human_segmentation(input_image, boxes, labels, scores, masks)

    return vis, masks