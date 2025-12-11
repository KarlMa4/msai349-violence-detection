import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset_sequence_loader import SequenceDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# -----------------------------------------------------
# ⚙️ Device Setup
# -----------------------------------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("🧠 Using device:", device)

# -----------------------------------------------------
# 🧩 Dataset + Transform
# -----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_ds = SequenceDataset("data/smart_city-sequences-5s/train", num_frames=16, transform=transform)
test_ds  = SequenceDataset("data/smart_city-sequences-5s/test",  num_frames=16, transform=transform)

# --- Merge "violence" and "weaponized" into a single label ---
merge_map = {
    "normal": 0,
    "violence": 1,
    "weaponized": 1
}

def remap_targets(dataset):
    new_targets = []
    for path, label_idx in dataset.samples:
        class_name = dataset.classes[label_idx]
        if class_name not in merge_map:
            raise ValueError(f"Unexpected class '{class_name}' in dataset.")
        new_targets.append(merge_map[class_name])
    dataset.targets = new_targets
    dataset.classes = ["normal", "violence"]
    return dataset

train_ds = remap_targets(train_ds)
test_ds  = remap_targets(test_ds)

print("✅ Loaded", len(train_ds), "training clips and", len(test_ds), "test clips.")
print("Classes after merge:", train_ds.classes)
print("Train label counts:", Counter(train_ds.targets))
print("Test label counts:", Counter(test_ds.targets))

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

# -----------------------------------------------------
# 🧱 Model Definition: CNN + LSTM
# -----------------------------------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1, num_classes=2):
        super(CNN_LSTM, self).__init__()

        base_model = models.resnet18(weights="IMAGENET1K_V1")
        modules = list(base_model.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.feature_dim = base_model.fc.in_features

        # freeze CNN initially
        for p in self.cnn.parameters():
            p.requires_grad = False

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=self.feature_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # 🔽 dropout before final classifier
        self.dropout = nn.Dropout(p=0.3)

        # Classification head
        self.fc = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x).squeeze(-1).squeeze(-1)  # [B*T, 512]
        feats = feats.view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        final_feat = self.dropout(lstm_out[:, -1, :])
        out = self.fc(final_feat)
        return out

# -----------------------------------------------------
# 🧩 Training Setup (regularized)
# -----------------------------------------------------
model = CNN_LSTM(hidden_size=256, num_layers=1, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,             # slightly higher LR
    weight_decay=1e-4    # regularization
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

EPOCHS = 25  # longer training

# -----------------------------------------------------
# 🔁 Training Loop
# -----------------------------------------------------
for epoch in range(EPOCHS):
    # Gradual unfreeze: unfreeze layer4 after 2 epochs
    if epoch == 2:
        for name, p in model.cnn.named_parameters():
            if "layer4" in name:
                p.requires_grad = True
        print("🔓 Unfroze layer4 for fine-tuning")

    model.train()
    running_loss, correct, total = 0, 0, 0

    for clips, labels in train_loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    acc = correct / total
    scheduler.step()
    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {running_loss/len(train_loader):.4f}  Accuracy: {acc:.3f}")

# -----------------------------------------------------
# 💾 Save model
# -----------------------------------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cnn_lstm_binary.pt")
print("✅ Model saved to models/cnn_lstm_binary.pt")

# -----------------------------------------------------
# 📊 Evaluation
# -----------------------------------------------------
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for clips, labels in test_loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        preds = outputs.argmax(1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

print("\n📊 Classification Report:\n")
print(classification_report(
    y_true, y_pred,
    labels=[0, 1],
    target_names=["normal", "violence"]
))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_ds.classes, yticklabels=train_ds.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("CNN + LSTM (Binary: Violence vs Non-Violence)")
plt.tight_layout()
plt.savefig("models/cnn_lstm_binary_confmat.png")
plt.show()
