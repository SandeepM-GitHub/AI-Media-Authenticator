import torch
from models.fusion_detector import FusionDetector

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
"""
“Why are you using a deprecated API?”
“I used PyTorch dynamic quantization for stability and compatibility with ONNX/TFLite. 
I’m aware of the new torchao APIs and would migrate 
when targeting PyTorch 2.10+, but for deployment-focused pipelines 
this approach is still widely used. Hence I have used warning filters to
suppress deprecation warnings.”
"""

# Load the pruned model
model = FusionDetector()
model.load_state_dict(
    torch.load("checkpoints/fusion_model_pruned.pt", 
    map_location='cpu'))

model.eval() # set to inference mode

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear},
    dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), 
    "checkpoints/fusion_model_quantized.pt"
)

print("Quantized model saved.")