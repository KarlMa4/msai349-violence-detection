"""
Comprehensive Evaluation of All 3 Models
=========================================
Evaluates R3D-18, MobileNet+LSTM, and VideoMAE on the test set.
Generates comparison graphs and performance metrics.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, roc_auc_score
)
from tqdm import tqdm
import time
import json

# Import model classes
from train_mobilenet_lstm import ViolenceDataset, MobileNetLSTM, HIDDEN_DIM, NUM_CLASSES
from torchvision.models.video import r3d_18
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# =====================================================================
# DATASET PREPARATION
# =====================================================================
class R3D18Dataset(torch.utils.data.Dataset):
    """Dataset for R3D-18 model"""
    def __init__(self, df, num_frames=16, resize_to=112):
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.resize_to = resize_to
        self.label_map = {"NonViolence": 0, "Violence": 1}

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
        npy_path = row["cleaned_file"]
        label = self.label_map[row["class"]]

        frames = np.load(npy_path)
        frames = self._sample_frames(frames)
        frames = frames.astype(np.float32) / 255.0

        # Apply transforms
        tensor_frames = []
        for f in frames:
            f = torch.from_numpy(f).permute(2, 0, 1)
            f = transforms.functional.resize(f, (self.resize_to, self.resize_to))
            tensor_frames.append(f)

        video = torch.stack(tensor_frames)
        video = video.permute(1, 0, 2, 3)  # (C,T,H,W)
        return video, label


class VideoMAEDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor, num_frames=16):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.num_frames = num_frames
        self.label_map = {"NonViolence": 0, "Violence": 1}

    def _sample_frames(self, frames):
        total = frames.shape[0]
        if total >= self.num_frames:
            idxs = np.linspace(0, total - 1, self.num_frames).astype(int)
            return [frames[i] for i in idxs]
        else:
            pad = [frames[-1]] * (self.num_frames - total)
            return list(frames) + pad

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        frames = np.load(row["cleaned_file"]).astype(np.uint8)
        sampled = self._sample_frames(frames)

        processed = self.processor(
            sampled,
            return_tensors="pt"
        )

        # correct: (1, C, T, H, W)
        pixel_values = processed["pixel_values"].squeeze(0)

        label = self.label_map[row["class"]]
        return {"pixel_values": pixel_values, "labels": label}

    def __len__(self):
        return len(self.df)


# =====================================================================
# MODEL EVALUATION FUNCTIONS
# =====================================================================
def evaluate_mobilenet(model, loader, criterion):
    """Evaluate MobileNet+LSTM model"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0

    with torch.no_grad():
        for clips, labels in tqdm(loader, desc="Evaluating MobileNet+LSTM", leave=False):
            clips, labels = clips.to(DEVICE), labels.to(DEVICE)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs), total_loss / len(loader.dataset)


def evaluate_r3d18(model, loader, criterion):
    """Evaluate R3D-18 model"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0

    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Evaluating R3D-18", leave=False):
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs), total_loss / len(loader.dataset)


def evaluate_videomae(model, loader):
    """Evaluate VideoMAE model"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating VideoMAE", leave=False):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs), total_loss / len(loader.dataset)


# =====================================================================
# METRICS CALCULATION
# =====================================================================
def calculate_metrics(preds, labels, probs):
    """Calculate comprehensive evaluation metrics"""
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    
    # ROC AUC (for binary classification)
    roc_auc = roc_auc_score(labels, probs[:, 1])
    
    cm = confusion_matrix(labels, preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }


# =====================================================================
# VISUALIZATION FUNCTIONS
# =====================================================================
def plot_confusion_matrices(results, figsize=(15, 4)):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        cm = metrics["metrics"]["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False)
        axes[idx].set_title(f"{model_name}\nConfusion Matrix")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_xticklabels(["NonViolence", "Violence"])
        axes[idx].set_yticklabels(["NonViolence", "Violence"])
    
    plt.tight_layout()
    plt.savefig("results/confusion_matrices.png", dpi=300, bbox_inches="tight")
    print("✅ Saved: results/confusion_matrices.png")
    plt.close()


def plot_metrics_comparison(results, figsize=(14, 5)):
    """Plot metrics comparison across models"""
    models = list(results.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    # Prepare data
    metrics_data = {metric: [] for metric in metrics_names}
    for model_name in models:
        for metric in metrics_names:
            metrics_data[metric].append(results[model_name]["metrics"][metric])
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    x = np.arange(len(models))
    width = 0.15
    for i, metric in enumerate(["accuracy", "f1", "precision", "recall"]):
        axes[0].bar(x + i*width, metrics_data[metric], width, label=metric.capitalize())
    
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Classification Metrics Comparison")
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Radar-like plot (individual model scores)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, model_name in enumerate(models):
        scores = [results[model_name]["metrics"][m] for m in metrics_names]
        axes[1].plot(metrics_names, scores, "o-", label=model_name, linewidth=2, markersize=8, color=colors[i])
    
    axes[1].set_ylabel("Score")
    axes[1].set_title("Metrics Profile by Model")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig("results/metrics_comparison.png", dpi=300, bbox_inches="tight")
    print("✅ Saved: results/metrics_comparison.png")
    plt.close()


def plot_roc_curves(results, figsize=(10, 8)):
    """Plot ROC curves for all models"""
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i, (model_name, model_results) in enumerate(results.items()):
        labels = model_results["labels"]
        probs = model_results["probs"]
        
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[i], lw=2.5, 
                label=f"{model_name} (AUC = {roc_auc:.3f})")
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/roc_curves.png", dpi=300, bbox_inches="tight")
    print("✅ Saved: results/roc_curves.png")
    plt.close()


def plot_speed_accuracy_tradeoff(results, figsize=(10, 7)):
    """Plot inference speed vs accuracy"""
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (model_name, model_results) in enumerate(results.items()):
        inference_time = model_results["inference_time_ms"]
        accuracy = model_results["metrics"]["accuracy"]
        
        ax.scatter(inference_time, accuracy, s=300, alpha=0.7, 
                  color=colors[i], label=model_name, edgecolors="black", linewidth=2)
        ax.annotate(model_name, (inference_time, accuracy), 
                   xytext=(10, 5), textcoords="offset points", fontsize=10, fontweight="bold")
    
    ax.set_xlabel("Inference Time per Sample (ms)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Speed vs Accuracy Trade-off", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.75, 1.0])
    
    plt.tight_layout()
    plt.savefig("results/speed_accuracy_tradeoff.png", dpi=300, bbox_inches="tight")
    print("✅ Saved: results/speed_accuracy_tradeoff.png")
    plt.close()

def plot_precision_recall_curves(results, figsize=(10, 7)):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    plt.figure(figsize=figsize)

    for name, r in results.items():
        labels = r["labels"]
        probs = r["probs"][:, 1]

        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)

        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})", linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/precision_recall_curves.png", dpi=300)
    print("✅ Saved: results/precision_recall_curves.png")
    plt.close()

def plot_class_wise_accuracy(results, figsize=(8, 6)):
    plt.figure(figsize=figsize)

    classes = ["NonViolence", "Violence"]
    idx = np.arange(len(classes))
    width = 0.35

    for i, (name, r) in enumerate(results.items()):
        preds = r["preds"]
        labels = r["labels"]

        accs = []
        for c in [0, 1]:
            mask = labels == c
            accs.append((preds[mask] == c).mean())

        plt.bar(idx + i*width, accs, width=width, label=name)

    plt.xticks(idx + width / 2, classes)
    plt.ylabel("Accuracy")
    plt.title("Class-wise Accuracy Comparison")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/class_wise_accuracy.png", dpi=300)
    print("✅ Saved: results/class_wise_accuracy.png")
    plt.close()


def print_summary_table(results):
    """Print comprehensive summary table"""
    print("\n" + "="*100)
    print("MODEL EVALUATION SUMMARY")
    print("="*100)
    
    summary_data = []
    for model_name, model_results in results.items():
        metrics = model_results["metrics"]
        summary_data.append({
            "Model": model_name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1-Score": f"{metrics['f1']:.4f}",
            "ROC-AUC": f"{metrics['roc_auc']:.4f}",
            "Loss": f"{model_results['loss']:.4f}",
            "Inference (ms)": f"{model_results['inference_time_ms']:.2f}",
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*100 + "\n")
    
    # Save to CSV
    summary_df.to_csv("results/evaluation_summary.csv", index=False)
    print("✅ Saved: results/evaluation_summary.csv")


# =====================================================================
# MAIN EVALUATION
# =====================================================================
def main():
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("🎬 VIOLENCE DETECTION - MODEL EVALUATION (NO VideoMAE)")
    print("="*80)
    
    # Load test dataset
    print("\n📂 Loading test dataset...")
    test_df = pd.read_csv("./data/cleaned_dataset/test_split.csv")
    print(f"✅ Loaded {len(test_df)} test samples")

    results = {}

    # ============================================================
    # 1. MOBILE NET + LSTM
    # ============================================================
    print("\n" + "-"*80)
    print("1️⃣  MOBILE NET + LSTM")
    print("-"*80)

    mobilenet_transform = transforms.Compose([
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                             std=[0.22803, 0.22145, 0.216989])
    ])
    
    mobilenet_loader = DataLoader(
        ViolenceDataset(test_df, mobilenet_transform),
        batch_size=4, shuffle=False
    )

    mobilenet_model = MobileNetLSTM(hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    mobilenet_model.load_state_dict(torch.load("models/mobilenet_lstm/model.pt", map_location=DEVICE))
    
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    preds_mb, labels_mb, probs_mb, loss_mb = evaluate_mobilenet(
        mobilenet_model, mobilenet_loader, criterion
    )
    inf_mb = (time.time() - start) / len(test_df) * 1000

    metrics_mb = calculate_metrics(preds_mb, labels_mb, probs_mb)

    results["MobileNet+LSTM"] = {
        "preds": preds_mb,
        "labels": labels_mb,
        "probs": probs_mb,
        "metrics": metrics_mb,
        "loss": loss_mb,
        "inference_time_ms": inf_mb
    }

    # ============================================================
    # 2. R3D-18
    # ============================================================
    print("\n" + "-"*80)
    print("2️⃣  R3D-18 (3D CNN)")
    print("-"*80)

    r3d_loader = DataLoader(
        R3D18Dataset(test_df),
        batch_size=4,
        shuffle=False
    )

    r3d18_model = r3d_18(weights="DEFAULT").to(DEVICE)
    r3d18_model.fc = nn.Linear(512, 2).to(DEVICE)
    r3d18_model.load_state_dict(torch.load("models/r3d18/model.pth", map_location=DEVICE))

    start = time.time()
    preds_r3d, labels_r3d, probs_r3d, loss_r3d = evaluate_r3d18(
        r3d18_model, r3d_loader, criterion
    )
    inf_r3d = (time.time() - start) / len(test_df) * 1000

    metrics_r3d = calculate_metrics(preds_r3d, labels_r3d, probs_r3d)

    results["R3D-18"] = {
        "preds": preds_r3d,
        "labels": labels_r3d,
        "probs": probs_r3d,
        "metrics": metrics_r3d,
        "loss": loss_r3d,
        "inference_time_ms": inf_r3d
    }

    # ============================================================
    # VISUALIZE
    # ============================================================
    plot_confusion_matrices(results)
    plot_metrics_comparison(results)
    plot_roc_curves(results)
    plot_speed_accuracy_tradeoff(results)
    plot_precision_recall_curves(results)
    plot_class_wise_accuracy(results)

    print_summary_table(results)


if __name__ == "__main__":
    main()
