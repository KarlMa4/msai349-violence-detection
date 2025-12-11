"""
Model Analysis — Save ALL misclassified videos as MP4
------------------------------------------------------
Instead of only showing videos using cv2.imshow, this script
saves every misclassified video into:

misclassified/
    false_positive/
    false_negative/

Also outputs summary CSV.
"""

import os
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms

from train_mobilenet_lstm import (
    ViolenceDataset,
    MobileNetLSTM,
    DEVICE,
    NUM_CLASSES,
    HIDDEN_DIM
)

SAVE_ROOT = "misclassified"


# ------------------------------------------------------------
# Save npy → mp4 utility
# ------------------------------------------------------------
def save_npy_as_mp4(npy_path, save_path, fps=10):
    video = np.load(npy_path)  # (T,H,W,3)
    T, H, W, C = video.shape

    # Convert frames to uint8 BGR for cv2
    frames = [(frame * 255).astype(np.uint8) for frame in video]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    for f in frames:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()


# ------------------------------------------------------------
@torch.no_grad()
def run_analysis():

    # Load test split
    test_df = pd.read_csv("./data/cleaned_dataset/test_split.csv")
    print(f"Loaded test set: {len(test_df)} samples")

    transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989]
        )
    ])

    ds = ViolenceDataset(test_df, transform)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # Load model
    model = MobileNetLSTM(hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("models/mobilenet_lstm/model.pt", map_location=DEVICE))
    model.eval()
    print("Loaded model ✓")

    y_true, y_pred, paths = [], [], []

    # --------------- Inference Loop ----------------
    for i, (clip, label) in enumerate(loader):
        clip = clip.to(DEVICE)
        out = model(clip)
        pred = out.argmax(1).item()

        y_true.append(label.item())
        y_pred.append(pred)
        paths.append(test_df.iloc[i]["cleaned_file"])

    # --------------- Compute Misclassifications ---------------
    false_pos, false_neg = [], []

    for i, (gt, pred) in enumerate(zip(y_true, y_pred)):
        p = paths[i]
        if gt == 0 and pred == 1:
            false_pos.append(p)
        elif gt == 1 and pred == 0:
            false_neg.append(p)

    print("\n❌ False Positives:", len(false_pos))
    print("❌ False Negatives:", len(false_neg))

    # --------------- Create Save Folders ---------------
    os.makedirs(f"{SAVE_ROOT}/false_positive", exist_ok=True)
    os.makedirs(f"{SAVE_ROOT}/false_negative", exist_ok=True)

    # --------------- Save Videos ---------------
    print("\nSaving misclassified videos...")

    summary = []

    # Save FP
    for i, fp in enumerate(false_pos):
        save_path = f"{SAVE_ROOT}/false_positive/fp_{i}.mp4"
        save_npy_as_mp4(fp, save_path)
        summary.append(["FP", fp, save_path])

    # Save FN
    for i, fn in enumerate(false_neg):
        save_path = f"{SAVE_ROOT}/false_negative/fn_{i}.mp4"
        save_npy_as_mp4(fn, save_path)
        summary.append(["FN", fn, save_path])

    # --------------- Save Summary CSV ---------------
    df = pd.DataFrame(summary, columns=["type", "source_path", "saved_mp4"])
    df.to_csv(f"{SAVE_ROOT}/misclassified_summary.csv", index=False)

    print("\n📁 Misclassification analysis complete!")
    print(f"All videos saved under: {SAVE_ROOT}/")
    print(f"Summary CSV saved at: {SAVE_ROOT}/misclassified_summary.csv")


if __name__ == "__main__":
    run_analysis()
