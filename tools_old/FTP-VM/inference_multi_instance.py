from argparse import ArgumentParser
from inference_model_list import inference_model_list
from model.which_model import get_model_by_string
import torch
from torch import nn
import glob
import cv2
import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from torch.nn import functional as F
from inference_io import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter

class ResizeShort(nn.Module):
    def __init__(self, short_size):
        super().__init__()
        self.short_size = short_size
    
    def forward(self, image):
        w, h = image.size
        if h < w:
            ratio = self.short_size / h
        else:
            ratio = self.short_size / w
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        if max(new_w, new_h) % 32 != 0:
            if new_w > new_h:
                new_w = ((new_w // 32) + 1) * 32
            else:
                new_h = ((new_h // 32) + 1) * 32
        return transforms.Resize((new_h, new_w))(image)

def convert_video(model,
                  input_source: str,
                  memory_img: str,
                  memory_mask: Optional[str] = None,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = torch.float32,
                  target_size: int = 1024):
    
    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """
    
    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'
    
    # Initialize transform
    transform = transforms.Compose([
        ResizeShort(576),
        transforms.ToTensor()
    ])

    # Initialize reader
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)
    

    # Initialize writers
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device

    # Load 1st frame and gen trimap from alpha
    m_img = transform(Image.open(memory_img)).unsqueeze(0).unsqueeze(0).to(device)

    if memory_mask is not None and memory_mask != '':
        m_mask = transform(Image.open(memory_mask).convert(mode='L')).unsqueeze(0).unsqueeze(0).to(device)
        unknown = m_mask[0,0,0].cpu().numpy()
        unknown = (unknown > 1.0/255.0) & (unknown < 254.0/ 255.9)
        unknown = cv2.dilate(unknown.astype('uint8'), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        unknown = unknown.astype('bool')
        unknown = torch.from_numpy(unknown).to(device)
        m_mask[0,0,0][unknown] = 0.5
        # import pdb; pdb.set_trace()
    else:
        print("Memory frame is background!")
        shape = list(m_img.shape) # b t c h w
        shape[2] = 1
        m_mask = torch.zeros(shape, dtype=m_img.dtype, device=m_img.device)
    bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    
    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            rec = model.default_rec
            memory = None
            for src, frame_names in reader:
                
                if downsample_ratio is None:
                    downsample_ratio = auto_downsample_ratio(*src.shape[2:], target=target_size)
                    print(downsample_ratio)
                if memory is None:
                    memory = model.encode_imgs_to_value(m_img, m_mask, downsample_ratio=downsample_ratio)

                src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
                
                trimap, matte, pha, rec = model.forward_with_memory(src, *memory, *rec, downsample_ratio=downsample_ratio)

                pha = pha.clamp(0, 1)
                trimap = seg_to_trimap(trimap)

                fgr = src * pha + bgr * (1 - pha)
                
                if output_foreground is not None:
                    writer_fgr.write(fgr[0])
                
                for i in range(len(frame_names)):
                    mask = trimap[0,i,0].cpu().numpy() > 0
                    mask = mask.astype('uint8') * 255
                    out_path = os.path.join(output_alpha.replace('ft_22k', 'mask'),frame_names[i].replace('.jpg', '/' + os.path.basename(memory_mask)))
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    cv2.imwrite(out_path, mask)

                if output_alpha is not None:
                    output_names = [x.replace('.jpg', "/" + os.path.basename(memory_mask)) for x in frame_names]
                    writer_pha.write(pha[0], output_names)

                if output_composition is not None:
                    # t, c, h, w
                    target_height = 540
                    rgb = torch.cat([src[0], fgr[0]], dim=3)
                    ratio = target_height / rgb.size(2)
                    rgb = F.interpolate(rgb, scale_factor=(ratio, ratio))

                    size = (rgb.size(-2), rgb.size(-1)//2)
                    # print(rgb.shape, size)
                    pha = F.interpolate(pha[0], size=size)# scale_factor=(ratio, ratio))

                    # ratio = target_height / trimap.size(3)
                    trimap = F.interpolate(trimap[0], size=size)#, scale_factor=(ratio, ratio))
                    
                    mask = torch.repeat_interleave(torch.cat([trimap, pha], dim=3), 3, dim=1)
                    w = min(rgb.size(-1), mask.size(-1))
                    dim = 2 if size[0] < size[1] else 3
                    out = torch.cat([rgb[..., :w], mask[..., :w]], dim=dim)
                    # print(out.shape)
                    writer_com.write(out)
                
                bar.update(src.size(1))

    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()

def seg_to_trimap(logit):
    val, idx = torch.sigmoid(logit).max(dim=2, keepdim=True) # ch
    # (bg, t, fg)
    tran_mask = idx == 1
    fg_mask = idx == 2
    return tran_mask*0.5 + fg_mask

def auto_downsample_ratio(h, w, target=1024):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 1024px.
    """
    # return min(target / min(h, w), 1)
    ratio = min(target / max(h, w), 1)
    print('auto size h, w = %f, %f' % (h*ratio, w*ratio))
    return ratio


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--input', help='input data root', required=True, type=str)
    parser.add_argument('--output', help='output data root', required=True, type=str)
    parser.add_argument('--gpu', help='gpu id', default=0, type=int)
    parser.add_argument('--seq_chunk', help='the frames to process in a batch', default=4, type=int)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

    root = args.input
    outroot = args.output
    model_name = 'FTPVM'
    os.makedirs(outroot, exist_ok=True)
    model_attr = inference_model_list[model_name]
    model = get_model_by_string(model_attr[1])().to(device='cuda')
    model_path = "/sensei-fs/users/chuongh/FTP-VM/FTP-VM/saves/Oct27_20.30.02_FTPVM_VIM_only/Oct27_20.30.02_FTPVM_VIM_only_2000.pth"
    # model.load_state_dict(torch.load(model_attr[3]))
    model.load_state_dict(torch.load(model_path))


    video_names = os.listdir(os.path.join(root, 'fgr'))
    for vid in video_names:
        input_source = os.path.join(root, 'fgr', vid)
        # Get first image and first alpha
        first_img_path = sorted(os.listdir(input_source))[0]
        first_img_path = os.path.join(input_source, first_img_path)
        all_alpha_paths = sorted(glob.glob(first_img_path.replace('fgr', 'pha').replace('.jpg', '/*.png')))
        output_name = os.path.join(outroot, vid)
        for i_inst, alpha_path in enumerate(all_alpha_paths):
            print("Processing %s - instance %d" % (vid, i_inst))
            convert_video(
                model,
                input_source=input_source,
                memory_img=first_img_path,
                memory_mask=alpha_path,
                output_type='png_sequence',
                output_composition = None,
                output_alpha = output_name,
                output_foreground = None,
                output_video_mbps=8,
                seq_chunk=args.seq_chunk,
                num_workers=0
            )