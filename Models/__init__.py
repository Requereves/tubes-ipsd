# Backward compatibility bridge to src.model
from src.model import build_model, CustomCNN

def get_model(name="regnet"):
    name_low = name.lower()
    if "regnet" in name_low:
        name_key = "regnet_y_800mf"
    elif "conv" in name_low:
        name_key = "convnext_tiny"
    else:
        name_key = "custom_cnn"
    return build_model(model_name=name_key, num_classes=6)

__all__ = ["get_model", "build_model", "CustomCNN"]
