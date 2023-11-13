import torch
from torch import nn
import time
import random

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            *[nn.Conv2d(3, 3, 3, 1, 1) for _ in range(30)])
        self.fc = nn.Linear(3, 3)
    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x

while True:
    idx = random.randint(0, 3)
    model = Model().to("cuda:{}".format(idx))
    x = torch.randn(1, 3, 256, 256).to("cuda:{}".format(idx))
    # time.sleep(10)
    x = model(x)
    del x, model
    torch.cuda.empty_cache()
    