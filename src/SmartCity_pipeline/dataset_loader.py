import torch
from torchvision import datasets, transforms
import platform
import os

class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom ImageFolder that also returns image file paths.
    Useful for evaluation when you need to know which file each prediction came from.
    """
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)  # (image_tensor, label)
        path = self.imgs[index][0]                  # image file path
        return original_tuple + (path,)


def get_dataloaders(train_dir="data/train_frames",
                    test_dir="data/test_frames",
                    batch_size=32,
                    num_workers=None):
    """
    Loads train/test image datasets and returns DataLoaders.
    Automatically disables multiprocessing on macOS and checks folder existence.
    """

    # ---- Step 0: Check folders ----
    for folder in [train_dir, test_dir]:
        if not os.path.exists(folder):
            print(f"⚠️ Folder not found: {folder}")
            print("➡️  Creating empty folder — please run 'extract_frames.py' to populate with .jpg frames.")
            os.makedirs(folder, exist_ok=True)

    # ---- Step 1. Define image transforms ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # match pretrained ResNet input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225])   # ImageNet std
    ])

    # ---- Step 2. Create datasets ----
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    test_ds  = datasets.ImageFolder(test_dir, transform=transform)

    # ---- Step 3. Handle num_workers automatically ----
    if num_workers is None:
        num_workers = 0 if platform.system() == "Darwin" else 2

    # ---- Step 4. Create DataLoaders ----
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # ---- Step 5. Helpful summary ----
    print(f"✅ Loaded {len(train_ds)} training and {len(test_ds)} testing images.")
    print(f"🧵 Using num_workers={num_workers}")
    if len(train_ds.classes) > 0:
        print(f"📂 Classes: {train_ds.classes}")
    else:
        print("⚠️ No images found! Make sure you ran frame extraction.")

    return train_loader, test_loader
