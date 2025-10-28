"""
Evaluate the SmartCity-trained baseline CNN
1️⃣  On SmartCity’s own test set  (within-domain performance)
2️⃣  On the RWF-2000 dataset       (cross-domain generalization)
Each evaluation prints a classification report and shows
a confusion matrix with accuracy / macro metrics overlay.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os
from dataset_loader import ImageFolderWithPaths

# -----------------------------------------------------------
# Common setup
# -----------------------------------------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print("🧠 Using device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def evaluate_dataset(model, dataloader, classes, title):
    """Run evaluation and plot confusion matrix with summary box."""
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    # --- metrics ---
    report_dict = classification_report(y_true, y_pred,
                                        target_names=classes,
                                        output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=classes)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n📊 {title}:\n")
    print(report_str)

    # --- confusion matrix plot ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    # summary box (accuracy + macro metrics)
    summary_text = (
        f"accuracy = {acc:.2f}\n"
        f"macro precision = {report_dict['macro avg']['precision']:.2f}\n"
        f"macro recall    = {report_dict['macro avg']['recall']:.2f}\n"
        f"macro f1-score  = {report_dict['macro avg']['f1-score']:.2f}"
    )
    plt.gcf().text(
        1.02, 0.5, summary_text,
        fontsize=11, va='center', ha='left',
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor='white', edgecolor='gray', alpha=0.9)
    )

    # per-class F1s (optional)
    class_metrics = "\n".join(
        [f"{cls:12s} F1 = {report_dict[cls]['f1-score']:.2f}"
         for cls in classes]
    )
    plt.gcf().text(
        1.02, 0.15, class_metrics,
        fontsize=10, va='center', ha='left',
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor='white', edgecolor='gray', alpha=0.9)
    )

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# Load model
# -----------------------------------------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("models/baseline_cnn.pt", map_location=device))
model = model.to(device)
model.eval()

# -----------------------------------------------------------
# 1️⃣ Evaluate on SmartCity test set
# -----------------------------------------------------------
smart_test = datasets.ImageFolder("data/smart_city-processed_image/test_frames", transform=transform)
smart_loader = torch.utils.data.DataLoader(smart_test, batch_size=32,
                                           shuffle=False, num_workers=0)
print(f"✅ SmartCity test images: {len(smart_test)} | Classes: {smart_test.classes}")

evaluate_dataset(model, smart_loader, smart_test.classes,
                 "SmartCity-trained Model on SmartCity Test Set")

# -----------------------------------------------------------
# 2️⃣ Evaluate on RWF-2000 test set
# -----------------------------------------------------------
rwf_test = datasets.ImageFolder("data/rwf2000-processed_image/test_frames",
                                transform=transform)
rwf_loader = torch.utils.data.DataLoader(rwf_test, batch_size=32,
                                         shuffle=False, num_workers=0)
print(f"✅ RWF-2000 test images: {len(rwf_test)} | Classes: {rwf_test.classes}")

evaluate_dataset(model, rwf_loader, rwf_test.classes,
                 "SmartCity-trained Model on RWF-2000 Dataset")
