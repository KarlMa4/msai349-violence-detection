import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# -----------------------------------------------------
# ⚙️ Setup
# -----------------------------------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("🧠 Using device:", device)

# -----------------------------------------------------
# 🧩 Data transforms with augmentation
# -----------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -----------------------------------------------------
# 🔹 Dataset wrapper to merge "weaponized" → "violence"
# -----------------------------------------------------
class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.class_to_idx = {"normal": 0, "violence": 1}

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        cls_name = self.base_dataset.classes[label]
        merged_label = 1 if cls_name in ["violence", "weaponized"] else 0
        return img, merged_label

    def __len__(self):
        return len(self.base_dataset)

# -----------------------------------------------------
# 📂 Load RWF dataset
# -----------------------------------------------------
rwf_train_dir = "data/rwf2000-processed_image/train_frames"
rwf_test_dir  = "data/rwf2000-processed_image/test_frames"

base_train = datasets.ImageFolder(rwf_train_dir, transform=train_transform)
base_test  = datasets.ImageFolder(rwf_test_dir, transform=test_transform)

train_ds = MergedDataset(base_train)
test_ds  = MergedDataset(base_test)

# Split fine-tune set (80% train / 20% val)
train_size = int(0.8 * len(train_ds))
val_size = len(train_ds) - train_size
train_subset, val_subset = random_split(train_ds, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

print(f"✅ Training: {len(train_subset)} | Validation: {len(val_subset)} | Testing: {len(test_ds)}")

# -----------------------------------------------------
# 🧱 Model setup
# -----------------------------------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/baseline_cnn_binary.pt", map_location=device))
model = model.to(device)

# Unfreeze layer3 + layer4 + fc for deeper fine-tuning
for name, param in model.named_parameters():
    if "layer2" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# -----------------------------------------------------
# 🧠 Early stopping parameters
# -----------------------------------------------------
best_val_loss = np.inf
patience, counter = 3, 0

# -----------------------------------------------------
# 🔁 Fine-tuning loop
# -----------------------------------------------------
EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct, total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = total_correct / total
    avg_train_loss = total_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}]  Train Loss: {avg_train_loss:.4f}  "
          f"Val Loss: {avg_val_loss:.4f}  Train Acc: {train_acc:.3f}  Val Acc: {val_acc:.3f}")

    # ---- Early stopping ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "models/fine_tuned_rwf_binary_best.pt")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("🛑 Early stopping triggered.")
            break

print("✅ Best fine-tuned model saved to models/fine_tuned_rwf_binary_best.pt")

# -----------------------------------------------------
# 📊 Evaluation
# -----------------------------------------------------
model.load_state_dict(torch.load("models/fine_tuned_rwf_binary_best.pt"))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

print("\n📊 Evaluation on RWF Test Set (Binary: Normal vs Violence):\n")
print(classification_report(y_true, y_pred, target_names=["normal", "violence"]))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["normal", "violence"],
            yticklabels=["normal", "violence"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Fine-tuned Binary Model (with Augmentation & Early Stopping)")
plt.tight_layout()
plt.show()
