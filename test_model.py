
import torch
from Models.efficientnet_b0 import create_model

# Tentukan device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Buat model
model = create_model(
    num_classes=6,
    pretrained=True
)

model = model.to(device)
model.eval()

# Simulasi 1 gambar RGB ukuran 224x224
dummy_image = torch.randn(1, 3, 224, 224).to(device)

# Prediksi
with torch.no_grad():
    output = model(dummy_image)
    probabilities = torch.softmax(output, dim=1)

print("Device:", device)
print("Output shape:", output.shape)
print("Probabilities:", probabilities)
print("Predicted class:", probabilities.argmax(dim=1).item())

assert output.shape == (1, 6)

print("MODEL BERHASIL DITES!")
