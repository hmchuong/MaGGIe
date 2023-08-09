import torch
from torch.utils.data import Dataset
try:
    from .video_matte import SingleInstComposedVidDataset
except ImportError:
    from video_matte import SingleInstComposedVidDataset

try:
    from .utils import gen_transition_gt
except ImportError:
    from utils import gen_transition_gt

class ComposedInstVidDataset():
    def __init__(self, max_inst=3, padding_inst=10,  **kwargs):
        super().__init__()
        self.ori_dataset = SingleInstComposedVidDataset(**kwargs)
        self.random = self.ori_dataset.random

        assert max_inst > 1, "Number of max instances should be greater than 1."
        self.max_inst = max_inst
        self.padding_inst = padding_inst
        
        # Filter data with more than one person
        split = kwargs['split']
        try:
            with open(f"vm2m/dataloader/valid_vm240k_{split}.txt") as f:
                valid_ids = set([line.strip() for line in f.readlines()])
        except:
            with open(f"valid_vm240k_{split}.txt") as f:
                valid_ids = set([line.strip() for line in f.readlines()])
        new_frame_ids = []
        for frame_name, start_idx in self.ori_dataset.frame_ids:
            if frame_name in valid_ids:
                new_frame_ids.append((frame_name, start_idx))
        self.ori_dataset.frame_ids = new_frame_ids
    
    def __len__(self):
        return len(self.ori_dataset)
    
    def denorm_image(self, image):
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        return image
    
    def __getitem__(self, idx):
        if self.max_inst > 2:
            more_frames = self.random.randint(1, self.max_inst - 1)
        else:
            more_frames = 1
        frame_idx = set(list(range(len(self.ori_dataset.frame_ids)))).difference(set([idx]))
        frame_idx = self.random.choice(list(frame_idx), more_frames, replace=False)
        frame_idx = [idx] + list(frame_idx)

        outputs = []
        for idx in frame_idx:
            outputs.append(self.ori_dataset.__getitem__(idx))
        
        # Combine image
        image = outputs[0]['image']
        image = self.denorm_image(image)

        for output in outputs[1:]:
            alpha = output['alpha']
            new_image = output['image']
            new_image = self.denorm_image(new_image)
            image = alpha * new_image + (1 - alpha) * image
        
        # Normalize image again
        image = (image - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # Combine alphas
        alphas = [outputs[0]['alpha']]
        for output in outputs[1:]:
            alpha = output['alpha']
            alphas.append(alpha)
            for j in range(len(alphas) - 1):
                alphas[j] = alphas[j] * (1 - alpha)

        alphas = torch.cat(alphas, dim=1)

        # Combine masks
        masks = [outputs[0]['mask']]
        for output in outputs[1:]:
            mask = output['mask']
            masks.append(mask)
            for j in range(len(masks) - 1):
                masks[j] = masks[j] * (1 - mask)

        masks = torch.cat(masks, dim=1)

        # Random place instances
        new_alphas = torch.zeros((alphas.shape[0], self.padding_inst, alphas.shape[2], alphas.shape[3]), dtype=torch.float32)
        new_masks = torch.zeros((masks.shape[0], self.padding_inst, masks.shape[2], masks.shape[3]), dtype=torch.float32)
        chosen_idx = self.random.choice(range(self.padding_inst), len(frame_idx), replace=False)
        new_alphas[:, chosen_idx] = alphas
        new_masks[:, chosen_idx] = masks
        alphas = new_alphas
        masks = new_masks

        # import pdb; pdb.set_trace()
        output_dict = {'image': image, 'mask': masks, 'alpha': alphas}

        # Generate transition GT from alphas
        k_size = self.random.choice(range(2, 5))
        iterations = self.random.randint(5, 15)
        
        transition_gt = gen_transition_gt(alphas.flatten(0, 1).unsqueeze(1), 
                                            masks.flatten(0, 1).unsqueeze(1), k_size, iterations)
        output_dict['transition'] = transition_gt.reshape(alphas.shape)

        return output_dict

if __name__ == "__main__":
    import cv2
    import numpy as np
    train_dataset = ComposedInstVidDataset(root_dir='/mnt/localssd/VideoMatte240K', split='train', clip_length=3, is_train=True, short_size=576,
                                            bg_dir='/mnt/localssd/bg', random_seed=2023, max_inst=3, padding_inst=10)
    
    for batch in train_dataset:
        frames = batch['image']
        alphas = batch['alpha']
        masks = batch['mask']
        transitions = batch['transition']
        for idx in range(len(frames)):
            frame = frames[idx]
            mask = masks[idx]
            alpha = alphas[idx]
            transition = transitions[idx]
            frame = frame * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            frame = (frame * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            cv2.imwrite("frame_{}.png".format(idx), frame[:, :, ::-1])
            for j in range(mask.shape[0]):
                cv2.imwrite("mask_{}_{}.png".format(idx, j), mask[j].numpy() * 255)
                cv2.imwrite("alpha_{}_{}.png".format(idx, j), alpha[j].numpy() * 255)
                cv2.imwrite("transition_{}_{}.png".format(idx, j), transition[j].numpy() * 255)
        import pdb; pdb.set_trace()