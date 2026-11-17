import torch.nn as nn
from torchvision import models
from torchvision.models import Swin_T_Weights


def create_model(num_classes=6, pretrained=True, freeze_backbone=False):

    weights = Swin_T_Weights.DEFAULT if pretrained else None

    model = models.swin_t(weights=weights)

    
    input_features = model.head.in_features
    model.head = nn.Linear(input_features, num_classes)

    
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    return model


if __name__ == "__main__":
    model = create_model(num_classes=6)
    print(model)
