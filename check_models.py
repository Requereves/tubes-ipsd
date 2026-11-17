import torch
from Models import get_model

for name in ["simple", "convnext", "resnet50", "efficientnet", "densenet"]:
    out = get_model(name)(torch.randn(2, 3, 224, 224))
    print(name, "OK", tuple(out.shape))