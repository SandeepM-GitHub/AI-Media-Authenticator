import torch
from torch.utils.data import DataLoader
from preprocessing.dataset import ImageDataset
from models.cnn_detector import CNNDetector

dataset = ImageDataset("data")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = CNNDetector()
loss_fn = torch.nn.CrossEntropyLoss()
# print(sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    total_loss = 0

    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")