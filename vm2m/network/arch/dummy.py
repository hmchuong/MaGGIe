from torch import nn

class Dummy(nn.Module):
    def __init__(self, backbone, decoder, cfg):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
    def forward(self, x, **kwargs):
        out = {"refined_masks": x["mask"]}
        return out