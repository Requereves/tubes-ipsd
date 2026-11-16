
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def create_model(num_classes=6, pretrained=True, freeze_backbone=False):

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None

    model = models.efficientnet_b0(weights=weights)

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        num_classes
    )

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    return model


if __name__ == "__main__":
    model = create_model(num_classes=6)
    print(model)
