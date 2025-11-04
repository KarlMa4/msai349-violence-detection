import os
from PIL import Image
from torch.utils.data import Dataset
import torch          
from torchvision import transforms
from torch.utils.data import DataLoader

"""
Step4: Dataset for loading sequences of frames from video clips.
"""
class SequenceDataset(Dataset):
    """
    Loads a sequence of frames from folders structured as:
        root/
          ├── normal/
          │     ├── clip_001/
          │     │     ├── frame_000.jpg ...
          ├── violence/
          │     ├── clip_042/ ...
    """

    def __init__(self, root_dir, num_frames=16, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        self.num_frames = num_frames
        self.transform = transform

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for clip_name in os.listdir(cls_dir):
                clip_dir = os.path.join(cls_dir, clip_name)
                if os.path.isdir(clip_dir):
                    self.samples.append((clip_dir, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_dir, label = self.samples[idx]
        frame_paths = sorted(os.listdir(clip_dir))
        frame_paths = [os.path.join(clip_dir, f) for f in frame_paths]

        # If clip has more than num_frames → sample evenly
        if len(frame_paths) >= self.num_frames:
            step = len(frame_paths) // self.num_frames
            frame_paths = frame_paths[::step][:self.num_frames]
        else:
            # Pad with last frame if fewer frames
            frame_paths += [frame_paths[-1]] * (self.num_frames - len(frame_paths))

        frames = []
        for fpath in frame_paths:
            img = Image.open(fpath).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Shape: [num_frames, 3, H, W]
        video_tensor = torch.stack(frames, dim=0)
        return video_tensor, label

"""
    Example usage
"""
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = SequenceDataset("data/smart_city-sequences-5s/train", num_frames=16, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

    clips, labels = next(iter(train_loader))
    print("✅ Dataset loaded successfully")
    print("Clips shape:", clips.shape)
    print("Labels:", labels)
