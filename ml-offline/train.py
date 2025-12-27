import torch
import os
from torch.utils.data import DataLoader
from preprocessing.dataset import ImageDataset
from models.fusion_detector import FusionDetector

dataset = ImageDataset("data")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = FusionDetector()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print("Trainable parameters:", 
    sum(p.numel() for p in model.parameters()))

for epoch in range(5):
    total_loss = 0.0

    for images, fft, labels in loader:
        optimizer.zero_grad()
        outputs = model(images, fft)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save the trained model
os.makedirs("checkpoints", exist_ok=True)

torch.save(model.state_dict(), "checkpoints/fusion_model.pt")
print("Fusion model saved to checkpoints/fusion_model.pt")