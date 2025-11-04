import os
import cv2
import glob
import argparse
from tqdm import tqdm
import numpy as np

def extract_sequence(video_path, output_dir, num_frames=8):
    """Extract N evenly spaced frames from a video and save as images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"⚠️ Empty video: {video_path}")
        return

    # pick evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(os.path.join(output_dir, clip_name), exist_ok=True)

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = os.path.join(output_dir, clip_name, f"frame_{i:03d}.jpg")
        cv2.imwrite(out_path, frame)

    cap.release()

def process_dataset(input_root, output_root, num_frames=8):
    classes = ["normal", "violence", "weaponized"]
    for cls in classes:
        input_dir = os.path.join(input_root, cls)
        output_dir = os.path.join(output_root, cls)
        os.makedirs(output_dir, exist_ok=True)

        video_files = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            video_files.extend(glob.glob(os.path.join(input_dir, ext)))

        print(f"📂 {cls}: {len(video_files)} videos found.")
        for vpath in tqdm(video_files, desc=f"Extracting {cls}"):
            extract_sequence(vpath, output_dir, num_frames=num_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract N frames per video for sequence modeling.")
    parser.add_argument("--input", required=True, help="Path to input dataset root (contains class folders).")
    parser.add_argument("--output", required=True, help="Path to output dataset root.")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames per video sequence.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    process_dataset(args.input, args.output, num_frames=args.frames)
