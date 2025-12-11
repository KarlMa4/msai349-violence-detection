"""
clean_dataset.py
----------------
Preprocess all videos for Violence Detection.

This script:
1. Loads videos from 'Violence' and 'NonViolence' folders.
2. Extracts 32 evenly spaced frames per video.
3. Resizes with letterboxing to 112x112 (preserve aspect ratio).
4. Converts frames to RGB and normalizes to [0, 1].
5. Saves each processed clip as a .npy file.
6. Records metadata to metadata.csv.

Run this after you've removed duplicate videos.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# CONFIGURATION
# -----------------------------
DATASET_DIR = "./data/Real_life_Violence_Dataset"
OUTPUT_DIR = "./data/cleaned_dataset"
SEQUENCE_LENGTH = 32       # number of frames to sample
TARGET_FPS = 15            # resample fps target (used for reference)
IMAGE_SIZE = (112, 112)    # (height, width)

# -----------------------------
# 1. LETTERBOX RESIZE FUNCTION
# -----------------------------
def letterbox(img, size=(112, 112), pad_value=0):
    """
    Resize image while preserving aspect ratio.
    Adds black padding if needed to fit target size.
    """
    h, w = img.shape[:2]
    oh, ow = size
    scale = min(oh / h, ow / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((oh, ow, 3), pad_value, dtype=np.uint8)
    top = (oh - nh) // 2
    left = (ow - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas

# -----------------------------
# 2. FRAME EXTRACTION FUNCTION
# -----------------------------
def extract_frames(video_path, num_frames=32, size=(112, 112)):
    """
    Uniformly sample 'num_frames' frames from a video.
    Returns a list of normalized RGB frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    duration = total / fps if fps > 0 else 0

    if total < num_frames:
        cap.release()
        return []

    # Choose frame indices evenly spaced
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize while keeping aspect ratio
        frame = letterbox(frame, size=size)

        # Normalize to 0–1 float
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    return frames if len(frames) == num_frames else []

# -----------------------------
# 3. MAIN CLEANING PIPELINE
# -----------------------------
def clean_dataset(root_dir, out_dir):
    """
    Process all videos and save cleaned .npy files + metadata.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_records = []

    for cls in ["Violence", "NonViolence"]:
        src_folder = Path(root_dir) / cls
        dst_folder = out_dir / cls
        dst_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n⚙️ Processing class: {cls}")
        video_files = list(src_folder.glob("*.mp4"))

        for video_path in tqdm(video_files):
            # Gather basic metadata
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Extract frames
            frames = extract_frames(video_path,
                                    num_frames=SEQUENCE_LENGTH,
                                    size=IMAGE_SIZE)

            if len(frames) != SEQUENCE_LENGTH:
                continue  # skip too-short or broken videos

            frames = np.asarray(frames, dtype=np.float32)

            # Save cleaned clip as .npy
            out_file = dst_folder / (video_path.stem + ".npy")
            np.save(out_file, frames)

            # Record metadata
            meta_records.append({
                "file": str(video_path),
                "cleaned_file": str(out_file),
                "class": cls,
                "fps": fps,
                "frames": total_frames,
                "width": width,
                "height": height,
                "duration_sec": total_frames / fps if fps else 0
            })

    # Save metadata to CSV
    df = pd.DataFrame(meta_records)
    df.to_csv(out_dir / "metadata.csv", index=False)
    print(f"\n✅ Cleaning complete! Saved {len(df)} cleaned clips and metadata.csv")

# -----------------------------
# 4. RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    clean_dataset(DATASET_DIR, OUTPUT_DIR)
