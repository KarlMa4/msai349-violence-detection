"""
Train R3D-18 3D CNN on npy video tensors
-----------------------------------------
Dataset structure:
cleaned_dataset/
    Violence/*.npy
    NonViolence/*.npy
"""

import os
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.video import r3d_18

from tqdm import tqdm


# -----------------------------------------------------------
# Device
# -----------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# -----------------------------------------------------------
# Dataset class for .npy videos
# -----------------------------------------------------------
class NPYVideoDataset(Dataset):
    def __init__(self, df, label2id, num_frames=16, resize_to=112, is_train=True):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.num_frames = num_frames
        self.resize_to = resize_to
        self.is_train = is_train

        self.frame_transform = T.Compose([
            T.Resize((resize_to, resize_to)),
            T.RandomHorizontalFlip() if is_train else (lambda x: x),
        ])

    def _sample_frames(self, frames):
        total = frames.shape[0]
        if total >= self.num_frames:
            idxs = np.linspace(0, total - 1, self.num_frames).astype(int)
            return frames[idxs]
        else:
            return np.concatenate(
                [frames, np.repeat(frames[-1][None], self.num_frames - total, axis=0)],
                axis=0
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        npy_path = row["video_path"]
        label_name = row["class"]
        label = self.label2id[label_name]

        frames = np.load(npy_path)          # shape (T,H,W,C)
        frames = self._sample_frames(frames)
        frames = frames.astype(np.float32) / 255.0

        # apply resize + flip per frame
        tensor_frames = []
        for f in frames:
            f = torch.from_numpy(f).permute(2,0,1)    # (C,H,W)
            f = self.frame_transform(f)
            tensor_frames.append(f)

        video = torch.stack(tensor_frames)  # (T,C,H,W)
        video = video.permute(1,0,2,3)      # -> (C,T,H,W) for 3D CNN

        return video, label


# -----------------------------------------------------------
# Training loop
# -----------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_correct = 0, 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(DEVICE), torch.tensor(y).to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x, y = x.to(DEVICE), torch.tensor(y).to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


# -----------------------------------------------------------
# Main training
# -----------------------------------------------------------
def main():
    root = Path("data/cleaned_dataset")

    # Load all npy paths
    rows = []
    for cls in ["Violence", "NonViolence"]:
        for f in (root / cls).iterdir():
            if f.suffix.lower() == ".npy":
                rows.append({"video_path": str(f), "class": cls})

    import pandas as pd
    df = pd.DataFrame(rows)
    print("Total videos:", len(df))

    # Split (70/15/15)
    trainval_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["class"], random_state=42
    )
    train_df, val_df = train_test_split(
        trainval_df, test_size=0.176, stratify=trainval_df["class"], random_state=42
    )

    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    label2id = {"NonViolence": 0, "Violence": 1}

    # Dataset
    train_ds = NPYVideoDataset(train_df, label2id, is_train=True)
    val_ds   = NPYVideoDataset(val_df,   label2id, is_train=False)
    test_ds  = NPYVideoDataset(test_df,  label2id, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=4, shuffle=False)

    # ---------------------------------------------------------
    # Model: R3D-18
    # ---------------------------------------------------------
    model = r3d_18(weights="DEFAULT")
    model.fc = nn.Linear(512, 2)      # 2 classes
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    for epoch in range(10):
        print(f"\nEpoch {epoch+1}/10")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc     = eval_one_epoch(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f}")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/r3d18/model.pth")
            print("⭐ Best model saved to models/r3d18/")

    # ---------------------------------------------------------
    # Final test evaluation
    # ---------------------------------------------------------
    model.load_state_dict(torch.load("models/r3d18/model.pth"))
    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion)

    print("\n🎯 Final Test Accuracy:", round(test_acc, 3))
    print("📉 Final Test Loss:", round(test_loss, 4))


if __name__ == "__main__":
    main()
