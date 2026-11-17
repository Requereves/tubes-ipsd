import torch.nn as nn
from torchvision import models

from .CNN import SimpleCNN
from .ConvNeXt import ConvNeXtTiny
from .efficientnet_b0 import create_model as create_efficientnet
from .mobilenet_v3 import create_model as create_mobilenet

NUM_CLASSES = 2  # GANTI sesuai jumlah kelas dataset-mu


# ---------- builder: semua model dibuat lewat sini ----------
def _simple(n):
    # CNN sederhana sebagai baseline (tanpa pretrained, input ukuran bebas)
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(128, n),
    )


def _cnn(n):
    # SimpleCNN dari CNN.py (input harus 128x128)
    return SimpleCNN(input_c=3, output=n)


def _convnext(n):
    return ConvNeXtTiny(output=n, pretrained=True)


def _efficientnet(n):
    return create_efficientnet(num_classes=n, pretrained=True)


def _mobilenet(n):
    return create_mobilenet(num_classes=n, pretrained=True)


def _resnet50(n):
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, n)
    return m


def _densenet(n):
    m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    m.classifier = nn.Linear(m.classifier.in_features, n)
    return m


# ---------- registry: model baru WAJIB didaftarkan di sini ----------
_REGISTRY = {
    "simple": _simple,
    "cnn": _cnn,
    "convnext": _convnext,
    "resnet50": _resnet50,
    "efficientnet": _efficientnet,
    "densenet": _densenet,
    "mobilenet": _mobilenet,
}


def get_model(name, num_classes=NUM_CLASSES):
    if name not in _REGISTRY:
        raise ValueError(f"Model '{name}' tidak dikenal. Pilihan: {list(_REGISTRY)}")
    return _REGISTRY[name](num_classes)