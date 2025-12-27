import torch
from models.fusion_detector import FusionDetector

model = FusionDetector()
model.load_state_dict(torch.load("checkpoints/fusion_model.pt"))
model.eval()

print("Fusion model loaded successfully")