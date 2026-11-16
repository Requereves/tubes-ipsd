
import torch
from Models.mobilenet_v3 import create_model

# Menentukan device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Membuat model
model = create_model(
    num_classes=6,
    pretrained=False
)

model = model.to(device)
model.eval()

# Simulasi gambar RGB 224x224
dummy_image = torch.randn(1, 3, 224, 224).to(device)

# Melakukan prediksi
with torch.no_grad():
    output = model(dummy_image)
    probabilities = torch.softmax(output, dim=1)

print("Model: MobileNetV3-Large")
print("Device:", device)
print("Output shape:", output.shape)
print("Probabilities:", probabilities)
print("Predicted class:", probabilities.argmax(dim=1).item())

assert output.shape == (1, 6)

print("MOBILENET BERHASIL DITES!")

