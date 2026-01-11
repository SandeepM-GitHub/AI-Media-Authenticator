import torch
from models.fusion_detector import FusionDetector

device = "cpu"

model = FusionDetector().to(device)
model.load_state_dict(
    torch.load("checkpoints/fusion_model.pt", map_location=device)
)
model.eval()

dummy_image = torch.randn(1, 3, 224, 224)
dummy_fft = torch.randn(1, 1024)

torch.onnx.export(
    model,
    (dummy_image, dummy_fft),
    "fusion_model.onnx",
    input_names=["images", "fft"],
    output_names=["logits"],
    opset_version=18,
    do_constant_folding=True
)

print("ONNX export successful")
