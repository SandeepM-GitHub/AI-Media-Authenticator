import torch
import os
from torch.utils.data import DataLoader, random_split
from preprocessing.dataset import ImageDataset
from models.fusion_detector import FusionDetector

# Create dataset
dataset = ImageDataset("data")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Split dataset into training and validation sets
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size]
)

# Create DataLoader for training and validation sets
train_loader = DataLoader(
    train_dataset, # Dataset object containing training data
    batch_size=2, # Number of samples per batch
    shuffle=True  # Randomly shuffle data at every epoch
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False # No need to shuffle validation data
)

# loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Initialize model, loss function, and optimizer
model = FusionDetector()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Validation loop
def evaluate(model, val_loader):
    model.eval() # inference mode
    correct = 0
    total = 0
    total_loss = 0.0
    

    with torch.no_grad(): # no gradients needed
        for images, fft, labels in val_loader:
            outputs = model(images, fft)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0) 
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0

    return avg_loss, accuracy

# Training loop
print("Trainable parameters:", 
    sum(p.numel() for p in model.parameters()))

for epoch in range(5):
    model.train() # training mode
    train_loss = 0.0

    for images, fft, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images, fft)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    print(
        f"Epoch {epoch+1} | "
        f" Train Loss: {train_loss:.4f} | "
        f" Val Loss: {val_loss:.4f} | "
        f" Val Accuracy: {val_acc*100:.2f}%"
    )



# Save the trained model
os.makedirs("checkpoints", exist_ok=True)

torch.save(model.state_dict(), "checkpoints/fusion_model.pt")
print("Fusion model saved to checkpoints/fusion_model.pt")