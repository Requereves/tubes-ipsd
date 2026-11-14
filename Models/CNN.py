import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_c=3, output=6):
        super(SimpleCNN, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )
        # input 128x128 -> 12x12 setelah 2 blok
        self.fc = nn.Linear(in_features=50 * 12 * 12, out_features=output)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        return self.fc(torch.flatten(x, 1))