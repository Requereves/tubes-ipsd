import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

def ConvNeXtTiny(output=6, pretrained=True):
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(768, output)
    return model