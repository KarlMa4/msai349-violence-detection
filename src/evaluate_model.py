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
import numpy as np

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
# 🧪 Comprehensive Evaluation Function
# -----------------------------------------------------
def evaluate_dataset(model, dataloader, title):
    """
    Comprehensive evaluation with multiple metrics:
    - Per-class metrics (Precision, Recall, F1)
    - Overall metrics (Accuracy, Macro/Weighted averages)
    - Confusion matrix visualization
    - Per-class breakdown tables
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, labels in dataloader:
            x, labels = x.to(device), labels.to(device)
            outputs = model(x)
            preds = outputs.argmax(1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    target_names = ["non_violence", "violence"]
    
    # Get detailed metrics
    report_dict = classification_report(
        y_true, y_pred, labels=[0, 1], 
        target_names=target_names, 
        output_dict=True
    )
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None
    )
    
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # ====== Print Header ======
    print(f"\n{'='*70}")
    print(f"📊 EVALUATION RESULTS: {title}")
    print(f"{'='*70}\n")

    # ====== Overall Metrics Table ======
    print(f"{'OVERALL METRICS':<40} {'Value':<15}")
    print(f"{'-'*55}")
    print(f"{'Accuracy':<40} {acc:.4f}")
    print(f"{'Macro Avg Precision':<40} {report_dict['macro avg']['precision']:.4f}")
    print(f"{'Macro Avg Recall':<40} {report_dict['macro avg']['recall']:.4f}")
    print(f"{'Macro Avg F1-Score':<40} {report_dict['macro avg']['f1-score']:.4f}")
    print(f"{'Weighted Avg Precision':<40} {report_dict['weighted avg']['precision']:.4f}")
    print(f"{'Weighted Avg Recall':<40} {report_dict['weighted avg']['recall']:.4f}")
    print(f"{'Weighted Avg F1-Score':<40} {report_dict['weighted avg']['f1-score']:.4f}")
    print()

    # ====== Per-Class Metrics Table ======
    print(f"{'PER-CLASS METRICS':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}")
    print(f"{'-'*70}")
    for i, class_name in enumerate(target_names):
        print(f"{class_name:<15} {precision[i]:<15.4f} {recall[i]:<15.4f} {f1[i]:<15.4f} {support[i]:<10}")
    print()

    # ====== Confusion Matrix Details ======
    print(f"{'CONFUSION MATRIX'}")
    print(f"{'-'*50}")
    print(f"                  Predicted Non-Violence  Predicted Violence")
    print(f"Actual Non-Violence       {cm[0,0]:<20}  {cm[0,1]:<10}")
    print(f"Actual Violence           {cm[1,0]:<20}  {cm[1,1]:<10}")
    print()

    # ====== Derived Metrics ======
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"{'DERIVED METRICS':<40} {'Value':<15}")
    print(f"{'-'*55}")
    print(f"{'Sensitivity (True Positive Rate)':<40} {sensitivity:.4f}")
    print(f"{'Specificity (True Negative Rate)':<40} {specificity:.4f}")
    print(f"{'False Positive Rate':<40} {(1-specificity):.4f}")
    print(f"{'False Negative Rate':<40} {(1-sensitivity):.4f}")
    print()

    # ====== Full Classification Report ======
    print(f"{'DETAILED CLASSIFICATION REPORT'}")
    print(f"{'-'*70}")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=target_names))

    # ====== Visualizations ======
    # Use GridSpec to reserve a bottom row for interpretation text so it won't be clipped
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 1], hspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis('off')

    # Confusion Matrix Heatmap - Enhanced
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create annotation text for each cell (count + row-percent)
    annotations = np.empty_like(cm, dtype=object)
    annotations[0, 0] = f'{tn}\n({cm_normalized[0,0]:.1%})'
    annotations[0, 1] = f'{fp}\n({cm_normalized[0,1]:.1%})'
    annotations[1, 0] = f'{fn}\n({cm_normalized[1,0]:.1%})'
    annotations[1, 1] = f'{tp}\n({cm_normalized[1,1]:.1%})'

    sns.heatmap(
        cm, annot=annotations, fmt='', cmap='RdYlGn', vmin=0, vmax=max(cm.max(), 1),
        xticklabels=['Predicted: Non-Violence', 'Predicted: Violence'],
        yticklabels=['Actual: Non-Violence', 'Actual: Violence'],
        ax=ax0, cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 11, 'fontweight': 'bold'},
        linewidths=2, linecolor='black'
    )
    ax0.set_xlabel("Predicted Label", fontsize=12, fontweight='bold')
    ax0.set_ylabel("Actual Label", fontsize=12, fontweight='bold')
    ax0.set_title("Confusion Matrix\n(Count + Percentage)", fontsize=12, fontweight='bold', pad=15)

    # Metrics Bar Chart
    metrics_data = {
        'Accuracy': acc,
        'Precision (NV)': precision[0],
        'Recall (NV)': recall[0],
        'F1 (NV)': f1[0],
        'Precision (V)': precision[1],
        'Recall (V)': recall[1],
        'F1 (V)': f1[1],
    }
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    bars = ax1.bar(range(len(metrics_data)), list(metrics_data.values()), color=colors, alpha=0.85)
    ax1.set_xticks(range(len(metrics_data)))
    ax1.set_xticklabels(metrics_data.keys(), rotation=45, ha='right')
    ax1.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax1.set_title("Performance Metrics", fontsize=12, fontweight='bold', pad=15)
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=9
        )

    # Add interpretation text in the reserved bottom row (full width)
    accuracy_text = (
        f"✓ True Negatives (TN): {tn} - Correctly identified non-violence\n"
        f"✗ False Positives (FP): {fp} - Non-violence labeled as violence\n"
        f"✗ False Negatives (FN): {fn} - Violence labeled as non-violence\n"
        f"✓ True Positives (TP): {tp} - Correctly identified violence"
    )
    ax_text.text(0.01, 0.5, accuracy_text, fontsize=11, family='monospace', va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm
    }

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
