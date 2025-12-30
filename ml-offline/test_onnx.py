import onnxruntime as ort
import numpy as np
import cv2

# Load the ONNX model
session = ort.InferenceSession(
    "checkpoints/fusion_model.onnx",
    providers=['CPUExecutionProvider']
)

# Print model inputs (for verification)
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)

# Load and preprocess an image
img = cv2.imread("data/ai/fish.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))  # Change to channel-first format CHW
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Dummy FFT input
fft = np.random.randn(1, 1024).astype(np.float32)

# Run inference
outputs = session.run(
    None,
    {
        "images": img,
        "fft": fft
    }
)

logits = outputs[0]
print("logits:", logits)
print("Prediction:", np.argmax(logits))

# Convert logits to probabilities using softmax
exp_logits = np.exp(logits - np.max(logits))
probabilities = exp_logits / np.sum(exp_logits)
print("Probabilities:", probabilities)
