"""
Train MobileNetV2 + LSTM for Violence Detection
------------------------------------------------
- Dataset: cleaned_dataset/ (each .npy = (32,112,112,3))
- Splits: Train 70%, Val 15%, Test 15% (test is saved for later)
"""

import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

# -----------------------------
# 1. CONFIG
# -----------------------------
DATA_CSV = "./data/cleaned_dataset/metadata.csv"
NUM_CLASSES = 2
SEQUENCE_LEN = 32
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 10
HIDDEN_DIM = 256
DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2. DATASET
# -----------------------------
class ViolenceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {"NonViolence": 0, "Violence": 1}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip = np.load(row["cleaned_file"])  # (T,H,W,3)
        frames = torch.tensor(clip, dtype=torch.float32).permute(0, 3, 1, 2)
        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])
        label = self.label_map[row["class"]]
        return frames, label

# -----------------------------
# 3. MODEL
# -----------------------------
class MobileNetLSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=2):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        base.classifier = nn.Identity()
        self.cnn = base.features
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.lstm = nn.LSTM(1280, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            f = self.cnn(x[:, t])
            f = self.gap(f).squeeze(-1).squeeze(-1)
            feats.append(f)
        feats = torch.stack(feats, dim=1)
        out, _ = self.lstm(feats)
        out = out[:, -1, :]
        return self.fc(out)

# -----------------------------
# 4. TRAIN / EVAL
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for clips, labels in tqdm(loader, leave=False):
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for clips, labels in loader:
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)
        outputs = model(clips)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

# -----------------------------
# 5. MAIN TRAINING LOOP
# -----------------------------
def main():
    df = pd.read_csv(DATA_CSV)

    # Split 70/15/15
    trainval_df, test_df = train_test_split(df, test_size=0.15, stratify=df["class"], random_state=42)
    train_df, val_df = train_test_split(trainval_df, test_size=0.176, stratify=trainval_df["class"], random_state=42)
    test_df.to_csv("./data/cleaned_dataset/test_split.csv", index=False)
    print("✅ Saved test split to ./data/cleaned_dataset/test_split.csv")

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                             std=[0.22803, 0.22145, 0.216989])
    ])

    train_loader = DataLoader(ViolenceDataset(train_df, transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ViolenceDataset(val_df, transform), batch_size=BATCH_SIZE, shuffle=False)

    model = MobileNetLSTM(HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val = 0
    print(f"Training on {DEVICE} | batch={BATCH_SIZE}, lr={LR}")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train {train_loss:.4f}/{train_acc:.3f} | Val {val_loss:.4f}/{val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "models/mobilenet_lstm/model.pt")

    print(f"✅ Training done. Best val acc: {best_val:.3f}")

if __name__ == "__main__":
    main()
