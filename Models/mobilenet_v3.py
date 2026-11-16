
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights


def create_model(num_classes=6, pretrained=False, freeze_backbone=False):

    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None

    model = models.mobilenet_v3_large(weights=weights)

    # Mengambil jumlah fitur classifier terakhir
    input_features = model.classifier[-1].in_features

    # Mengganti output menjadi 6 kelas
    model.classifier[-1] = nn.Linear(
        input_features,
        num_classes
    )

    # Opsional membekukan feature extractor
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    return model


if __name__ == "__main__":
    model = create_model(num_classes=6)
    print(model)
