import torch
import torch.nn as nn
from torchvision import models
from dataset_loader import get_dataloaders

# --- Step 1: Set up device (GPU / MPS / CPU) ---
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("🖥️ Using device:", device)

# --- Step 2: Load our training & test data ---
train_loader, test_loader = get_dataloaders(num_workers=0)

# --- Step 3: Load a pretrained CNN ---
# ResNet-18 is small but powerful enough for this baseline.
model = models.resnet18(weights="IMAGENET1K_V1")
# Replace the final layer to match our 3 classes
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

# --- Step 4: Define loss function and optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Step 5: Train the model ---
epochs = 5  # start small, we can increase later
for epoch in range(epochs):
    model.train()
    total, correct, running_loss = 0, 0, 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Accuracy: {acc:.3f}")

# --- Step 6: Save model weights ---
torch.save(model.state_dict(), "models/baseline_cnn.pt")
print("✅ Model saved to models/baseline_cnn.pt")
