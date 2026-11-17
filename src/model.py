import torch
import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    """
    Arsitektur CNN kustom mandiri (non-pretrained baseline).
    Menggunakan AdaptiveAvgPool2d agar fleksibel terhadap berbagai resolusi input gambar
    serta dilengkapi BatchNorm2d dan Dropout untuk mereduksi overfitting.
    """
    def __init__(self, in_channels=3, num_classes=6):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(model_name="regnet_y_800mf", num_classes=6, pretrained=True):
    """
    Model Factory untuk menginisialisasi model deep learning.
    Pilihan model:
    - 'regnet_y_800mf': Arsitektur RegNet (AnyNet space) dari Meta AI, akurat, cepat & jarang dipakai kompetitor.
    - 'convnext_tiny': Arsitektur modern ConvNeXt dari Meta AI
    - 'efficientnet_b0': Model efisien dengan compound scaling
    - 'custom_cnn': Arsitektur CNN mandiri yang dirancang khusus
    """
    model_key = model_name.lower().replace("-", "_").replace(" ", "")

    if "regnet" in model_key:
        weights = models.RegNet_Y_800MF_Weights.DEFAULT if pretrained else None
        model = models.regnet_y_800mf(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif "convnext" in model_key:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    elif "efficientnet" in model_key:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    elif model_key in ["custom_cnn", "simple", "cnn"]:
        return CustomCNN(in_channels=3, num_classes=num_classes)

    else:
        raise ValueError(
            f"Model '{model_name}' tidak dikenali. Pilih: ['convnext_tiny', 'efficientnet_b0', 'custom_cnn']"
        )
