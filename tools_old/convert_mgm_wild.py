import torch

state_dict = torch.load('pretrain/wild_matting.pth', map_location=torch.device('cpu'))

state_dict = state_dict['state_dict']

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module.backbone"):
        new_k = k.replace("module.backbone.", "")
        # if new_k.startswith("encoder."):
        #     import pdb; pdb.set_trace()
        #     new_k = new_k.replace("encoder.", "backbone.")
        new_state_dict[new_k] = v

torch.save(new_state_dict, 'pretrain/wild_matting_converted.pth')
# import pdb; pdb.set_trace()