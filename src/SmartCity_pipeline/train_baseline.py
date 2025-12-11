import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("🧠 Using device:", device)

# --- Transform (same as before) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --- Custom dataset wrapper to merge classes ---
class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.class_to_idx = {"normal": 0, "violence": 1}  # merged labels

    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        class_name = self.base_dataset.classes[label]
        merged_label = 1 if class_name in ["violence", "weaponized"] else 0
        return img, merged_label

    def __len__(self):
        return len(self.base_dataset)

# --- Load and merge dataset labels ---
train_base = datasets.ImageFolder("data/smart_city-processed_image/train_frames", transform=transform)
test_base  = datasets.ImageFolder("data/smart_city-processed_image/test_frames", transform=transform)

train_ds = MergedDataset(train_base)
test_ds  = MergedDataset(test_base)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

print(f"✅ Loaded {len(train_ds)} training and {len(test_ds)} testing images.")
print("Merged classes:", train_ds.class_to_idx)

# --- Model ---
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 output classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Train ---
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total, correct, running_loss = 0, 0, 0
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
    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {running_loss/len(train_loader):.4f}  Acc: {acc:.3f}")

# --- Save model ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/baseline_cnn_binary.pt")
print("✅ Model saved as baseline_cnn_binary.pt")
