import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN branch for image features
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.cnn_fc = nn.Linear(32 * 56 * 56, 128)  # Assuming input images are 224x224

        # FFT branch
        self.fft_fc = nn.Linear(32 * 32, 128)

        # Fusion 
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, image, fft):
        cnn_feat = self.cnn_fc(self.cnn_branch(image))
        fft_feat = self.fft_fc(fft)

        fused = torch.cat([cnn_feat, fft_feat], dim=1) 

        return self.classifier(fused)
