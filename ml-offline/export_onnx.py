import torch
from models.fusion_detector import FusionDetector

# Load the model architecture
model = FusionDetector()
model.load_state_dict(
    torch.load("checkpoints/fusion_model_pruned.pt", map_location=('cpu'))
)

# Set the model to evaluation mode
model.eval()

# Create dummy inputs for the ONNX export
dummy_images = torch.randn(1, 3, 224, 224)
dummy_fft = torch.randn(1, 1024)

# Export the model to ONNX format
torch.onnx.export(
    model,
    (dummy_images, dummy_fft),
    "checkpoints/fusion_model.onnx",
    input_names=["images", "fft"],
    output_names=["output"],
    opset_version=13,
)

print("Model has been successfully exported to ONNX format.")