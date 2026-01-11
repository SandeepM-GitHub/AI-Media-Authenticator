import torch
import torch.nn as nn
import torchvision.models as models

class FusionDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # --- CNN branch (suppressed) ---
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.cnn_out = 512

        # Strong compression → weak influence
        self.cnn_fc = nn.Sequential(
            nn.Linear(self.cnn_out, 32),
            nn.ReLU()
        )

        # --- FFT branch (dominant) ---
        self.fft_fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # --- Fusion classifier (FFT-heavy) ---
        self.classifier = nn.Sequential(
            nn.Linear(32 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, images, fft):
        # CNN
        x = self.cnn(images)
        x = x.view(x.size(0), -1)
        x = self.cnn_fc(x)

        # FFT
        f = self.fft_fc(fft)

        # Fuse
        fused = torch.cat([x, f], dim=1)
        return self.classifier(fused)
