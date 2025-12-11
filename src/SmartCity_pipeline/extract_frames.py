import os, cv2, glob
from tqdm import tqdm

def extract_middle_frame(video_path, output_dir):
    """Extracts the middle frame from a video and saves it as .jpg"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"⚠️ Skipping empty video: {video_path}")
        return
    mid_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    success, frame = cap.read()
    if success:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir,
                                os.path.basename(video_path).rsplit('.', 1)[0] + '.jpg')
        cv2.imwrite(out_path, frame)
    cap.release()


def process_folder(in_dir, out_dir):
    """Process all videos inside a directory (by class subfolders)."""
    classes = ["normal", "violence", "weaponized"]
    for cls in classes:
        input_folder = os.path.join(in_dir, cls)
        output_folder = os.path.join(out_dir, cls)

        if not os.path.exists(input_folder):
            print(f"⚠️ Missing input folder: {input_folder}")
            continue

        videos = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            videos.extend(glob.glob(os.path.join(input_folder, ext)))

        if len(videos) == 0:
            print(f"⚠️ No videos found in {input_folder}")
            continue

        print(f"📂 Found {len(videos)} videos in class '{cls}'")
        for v in tqdm(videos, desc=f"Extracting {cls}"):
            extract_middle_frame(v, output_folder)


if __name__ == "__main__":
    # Process both train and test
    input_folder = input("Enter input folder name (e.g., '1sec_processed'): ").strip()
    folder_name = input("Enter folder name to store images(e.g., 'smart_city-processed'): ").strip()
    process_folder(f"data/{input_folder}/Train", f"data/{folder_name}/train_frames")
    process_folder(f"data/{input_folder}/Test", f"data/{folder_name}/test_frames")
