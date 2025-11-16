"""
Evaluate violence-detection models (baseline CNN, CNN+LSTM).

Usage examples:
    python3 src/evaluate_model.py --model cnn       --weights models/baseline_cnn_binary.pt --dataset smartcity
    python3 src/evaluate_model.py --model cnn_lstm  --weights models/cnn_lstm_binary.pt    --dataset smartcity
    python3 src/evaluate_model.py --model cnn       --weights models/fine_tuned_rwf_binary.pt --dataset rwf2000

The script:
1️⃣ Loads the selected model architecture and checkpoint
2️⃣ Loads the matching dataset (ImageFolder or SequenceDataset)
3️⃣ Merges 'weaponized' → 'violence' for 2-class evaluation
4️⃣ Prints classification report and plots confusion matrix
"""

import argparse
import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# ⚙️ Device
# -----------------------------------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("🧠 Using device:", device)

# -----------------------------------------------------
# 🔧 Transforms
# -----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -----------------------------------------------------
# 📂 Merge helper (for ImageFolder datasets)
# -----------------------------------------------------
class MergedDataset(torch.utils.data.Dataset):
    """Wrap an ImageFolder dataset and merge 'weaponized'→'violence'."""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.classes = ["non_violence", "violence"]

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        cls_name = self.base_dataset.classes[label]
        merged_label = 1 if cls_name in ["violence", "weaponized"] else 0
        return img, merged_label

    def __len__(self):
        return len(self.base_dataset)

# -----------------------------------------------------
# 🧱 Model factory
# -----------------------------------------------------
def build_model(model_type: str, num_classes: int = 2):
    """Return the requested model architecture."""
    if model_type == "cnn":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "cnn_lstm":
        from dataset_sequence_loader import SequenceDataset

        class CNN_LSTM(nn.Module):
            def __init__(self, hidden_size=256, num_layers=1, num_classes=2):
                super().__init__()
                base_model = models.resnet18(weights=None)
                self.cnn = nn.Sequential(*list(base_model.children())[:-1])
                self.feature_dim = base_model.fc.in_features
                self.lstm = nn.LSTM(
                    input_size=self.feature_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                )
                self.dropout = nn.Dropout(p=0.3)
                self.fc = nn.Linear(hidden_size, num_classes)

            def forward(self, x):  # x: [B, T, 3, H, W]
                B, T, C, H, W = x.shape
                x = x.view(B * T, C, H, W)
                feats = self.cnn(x).squeeze(-1).squeeze(-1)
                feats = feats.view(B, T, -1)
                lstm_out, _ = self.lstm(feats)
                final_feat = self.dropout(lstm_out[:, -1, :])
                return self.fc(final_feat)

        model = CNN_LSTM(num_classes=num_classes)

    else:
        raise ValueError(f"❌ Unknown model type: {model_type}")

    return model

# -----------------------------------------------------
# 🧪 Evaluation function
# -----------------------------------------------------
def evaluate_dataset(model, dataloader, title):
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, labels in dataloader:
            x, labels = x.to(device), labels.to(device)
            outputs = model(x)
            preds = outputs.argmax(1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    target_names = ["non_violence", "violence"]
    report_dict = classification_report(y_true, y_pred,labels=[0, 1],target_names=["non_violence", "violence"],output_dict=True)
    acc = accuracy_score(y_true, y_pred)

    print(f"\n📊 {title}\n")
    print(classification_report(y_true, y_pred,labels=[0, 1],target_names=["non_violence", "violence"]))

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    summary_text = (
        f"accuracy = {acc:.2f}\n"
        f"macro precision = {report_dict['macro avg']['precision']:.2f}\n"
        f"macro recall    = {report_dict['macro avg']['recall']:.2f}\n"
        f"macro f1-score  = {report_dict['macro avg']['f1-score']:.2f}"
    )
    plt.gcf().text(1.02, 0.5, summary_text, fontsize=11, va='center', ha='left',
                   bbox=dict(boxstyle="round,pad=0.4",
                             facecolor='white', edgecolor='gray', alpha=0.9))
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------

# -----------------------------------------------------
# 🚀 Main
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "cnn_lstm"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", choices=["smartcity", "rwf2000"], required=True)
    args = parser.parse_args()

    # -------------------------------------------------
    # 📂 Resolve dataset path based on selection
    # -------------------------------------------------
    if args.dataset == "smartcity":
        if args.model == "cnn":
            path = "data/smart_city-processed_image/test_frames"
        elif args.model == "cnn_lstm":
            path = "data/smart_city-sequences-5s/test"
        else:
            raise ValueError("❌ Unknown model type for smartcity")

    elif args.dataset == "rwf2000":
        if args.model == "cnn":
            path = "data/rwf2000-processed_image/test_frames"
        elif args.model == "cnn_lstm":
            path = "data/smart_city-sequences-5s/test"
        else:
            raise ValueError("❌ Unknown model type for rwf2000")

    else:
        raise ValueError(f"❌ Unknown dataset: {args.dataset}")

    # -------------------------------------------------
    # 🧩 Load dataset
    # -------------------------------------------------
    if args.model == "cnn":
        base = datasets.ImageFolder(path, transform=transform)
        ds = MergedDataset(base)

    elif args.model == "cnn_lstm":
        from dataset_sequence_loader import SequenceDataset
        ds = SequenceDataset(path, num_frames=16, transform=transform)

    else:
        raise ValueError("❌ Unknown model type.")

    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    print(f"✅ Loaded {len(ds)} samples from {path}")

    # -------------------------------------------------
    # 🧠 Load model + weights
    # -------------------------------------------------
    model = build_model(args.model, num_classes=2)
    weights = torch.load(args.weights, map_location=device)

    # Handle checkpoints with extra keys (e.g., from training script)
    if isinstance(weights, dict) and "model_state" in weights:
        model.load_state_dict(weights["model_state"])
    else:
        model.load_state_dict(weights)

    model.to(device)
    model.eval()
    print(f"✅ Loaded weights from {args.weights}")

    # -------------------------------------------------
    # 📊 Evaluate model
    # -------------------------------------------------
    title = f"{args.model.upper()} on {args.dataset.upper()} test set"
    evaluate_dataset(model, loader, title)
