import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------------------
# ⚙️ Setup
# -----------------------------------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("🧠 Using device:", device)

# --- Transform (same as baseline)
transform = transforms.Compose([
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
# 📂 Load RWF dataset (for fine-tuning)
# -----------------------------------------------------
rwf_train_dir = "data/rwf2000-processed_image/train_frames"
rwf_test_dir  = "data/rwf2000-processed_image/test_frames"

base_train = datasets.ImageFolder(rwf_train_dir, transform=transform)
base_test  = datasets.ImageFolder(rwf_test_dir, transform=transform)

train_ds = MergedDataset(base_train)
test_ds  = MergedDataset(base_test)

# Optional: fine-tune on a portion of RWF
fine_tune_size = int(0.5 * len(train_ds))  # use 50% for fine-tuning
fine_tune_ds, _ = random_split(train_ds, [fine_tune_size, len(train_ds) - fine_tune_size])

train_loader = DataLoader(fine_tune_ds, batch_size=32, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

print(f"✅ Using {len(fine_tune_ds)} images for fine-tuning and {len(test_ds)} for testing.")
print("Merged classes:", train_ds.class_to_idx)

# -----------------------------------------------------
# 🧱 Model Setup
# -----------------------------------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2-class output
model.load_state_dict(torch.load("models/baseline_cnn_binary.pt", map_location=device))
model = model.to(device)

# Freeze early layers (keep learned low-level filters)
for name, param in model.named_parameters():
    if "layer3" not in name and "layer4" not in name and "fc" not in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

# -----------------------------------------------------
# 🔁 Fine-tuning loop
# -----------------------------------------------------
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {running_loss/len(train_loader):.4f}  Accuracy: {acc:.3f}")

# Save fine-tuned model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/fine_tuned_rwf_binary.pt")
print("✅ Fine-tuned binary model saved to models/fine_tuned_rwf_binary.pt")

# -----------------------------------------------------
# 📊 Evaluation
# -----------------------------------------------------
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
plt.title("Fine-tuned Binary Model on RWF-2000 Test Set")
plt.tight_layout()
plt.show()
