# 🎥 Violence Detection Using CCTV (MSAI 349 – Group 9)

**Team Members:**  
- Chia-Lu Karl Ma (lro6877)  
- Jamal Moussa (xrz2515)  
- Ramakrishna Saravanan (cqp0132)

---

## 🧩 Project Summary

This project explores **violence detection from CCTV videos** using deep learning models.

We started with the **SmartCity CCTV Violence Detection (SCVD)** dataset — a small, synthetic dataset containing short clips labeled as:
- `normal`
- `violence`
- `weaponized`

From there, we built and compared several models to understand how well deep learning can identify violent behavior.

---

## 📊 Early Experiments on SmartCity Dataset

### Dataset Overview and Label Consolidation
The SmartCity dataset contains ~400 short clips categorized into three classes: **normal**, **violence**, and **weaponized**. We merged violence and weaponized into a single **violence class**, resulting in **binary classification (violence vs. non-violence)**. This adjustment mitigated class imbalance and enabled fair comparison across models.

---

### Baseline Spatial Model: ResNet-18 on Single Frames

| Experiment | Accuracy | Observations |
|:-----------|:---------|:-------------|
| **Baseline on SmartCity (original split)** | 100% ❌ | Suspiciously perfect — suggested data leakage |
| **Same model on RWF-2000 (external test)** | ~50% | Confirmed train-test contamination |
| **After pHash duplicate removal** | 52.4% | True performance; spatial features insufficient |

**Key Finding:** The initial 100% accuracy was an artifact of **duplicate/near-duplicate clips** between training and test sets. After cleaning, accuracy dropped dramatically, revealing that spatial features alone cannot reliably detect violence.

---

### Duplicate Detection and Dataset Cleaning

We developed a **perceptual hash (pHash) duplicate detection tool** to identify and remove identical or visually similar frames across both sets. This cleaning process:
- ✅ Eliminated dataset contamination
- ✅ Revealed the true model capability (~52% accuracy)
- ✅ Validated the need for better data and temporal modeling

---

### Cross-Dataset Fine-Tuning: RWF-2000 → SmartCity

We fine-tuned the baseline ResNet-18 using a small subset of **RWF-2000** videos to test transfer learning:
- **Result:** Marginal improvement to ~55% accuracy
- **Insight:** Limited gains indicated:
  - Partial visual alignment between datasets
  - RWF-2000 subset too small to offset SmartCity's scarcity
  - Intrinsic dataset inconsistencies limited transfer learning benefits

**Takeaway:** External data helps, but only modestly without a larger, more coherent primary dataset.

---

### Temporal Modeling: CNN + LSTM

After reaching the spatial modeling ceiling, we extended the architecture with **temporal features**:
- **Architecture:** ResNet-18 (CNN) + LSTM over 8–16 evenly spaced frames
- **Result:** **38% accuracy** on SmartCity test set
- **Issue:** Unstable precision/recall; model failed to learn meaningful temporal dependencies

**Why it failed:** SmartCity's small size (~400 clips), short clip duration, and inconsistent motion patterns were insufficient for LSTM training.

---

## 🧠 What We Learned

1. **SmartCity data contained duplicates** — revealed by pHash tool; cleaned split showed true baseline at ~52% accuracy.
2. **Spatial features alone are insufficient** — single-frame ResNet-18 plateaued at 52.4%, even after data cleaning.
3. **Temporal modeling needs more data** — CNN + LSTM dropped to 38%, indicating the dataset was too small for motion learning.
4. **Transfer learning has limits** — RWF-2000 fine-tuning improved results only marginally (~3%), confirming dataset scarcity was the core bottleneck.
5. **Real-world datasets are essential** — SmartCity's synthetic nature, limited diversity, and small size made it unsuitable for robust model training.

---

## � Transition to Real-Life Violence Dataset (RLVD)

The SmartCity experiments revealed critical limitations:
- **Data contamination** (duplicate clips between train/test)
- **Insufficient scale** (~400 clips for spatial + temporal learning)
- **Synthetic nature** (unrepresentative of real CCTV scenarios)

We have transitioned to the **Real-Life Violence Dataset (RLVD)**, which contains:
- ✅ **~2,000 high-quality `.mp4` videos**
- ✅ **Balanced class distribution** (violence vs. non-violence)
- ✅ **Diverse real-world scenarios** (indoor, outdoor, various lighting/angles)
- ✅ **Consistent temporal structure** (longer clips suitable for motion modeling)

This dataset forms the foundation for **robust spatiotemporal violence detection**, enabling:
- Stronger transfer learning via pre-trained models
- Meaningful temporal learning (3D CNNs, Bi-LSTMs, Transformers)
- Improved domain generalization to real CCTV footage

---

## 📂 Repository Structure

```
msai349-violence-detection/
├── src/
│   ├── train_baseline.py              # Train ResNet-18 on single frames
│   ├── train_cnn_lstm.py              # Train CNN + LSTM on frame sequences
│   ├── evaluate_model.py              # Comprehensive evaluation (metrics, plots, confusion matrix)
│   ├── extract_frames.py              # Extract middle frame from videos
│   ├── extract_sequences.py           # Extract N evenly-spaced frames per video
│   ├── dataset_sequence_loader.py     # PyTorch Dataset for frame sequences
│   └── check_cross_dataset_duplicates.py  # pHash duplicate detection tool
├── models/
│   ├── baseline_cnn_binary.pt         # Trained ResNet-18
│   ├── cnn_lstm_binary.pt             # Trained CNN + LSTM
│   └── fine_tuned_rwf_binary.pt       # Fine-tuned on RWF-2000
├── data/
│   ├── smart_city-processed_image/    # SmartCity (frames)
│   ├── rwf2000-processed_image/       # RWF-2000 (frames) — for fine-tuning
│   └── ...
└── README.md
```

---

## 🚀 How to Run Evaluation

```bash
# Evaluate baseline CNN on SmartCity
python3 src/evaluate_model.py --model cnn --weights models/baseline_cnn_binary.pt --dataset smartcity

# Evaluate CNN + LSTM on SmartCity  
python3 src/evaluate_model.py --model cnn_lstm --weights models/cnn_lstm_binary.pt --dataset smartcity

# Evaluate fine-tuned CNN on RWF-2000
python3 src/evaluate_model.py --model cnn --weights models/fine_tuned_rwf_binary.pt --dataset rwf2000
```

The evaluation script provides:
- **Overall metrics:** Accuracy, Macro/Weighted Precision, Recall, F1
- **Per-class metrics:** Individual precision, recall, F1, support for each class
- **Derived metrics:** Sensitivity, Specificity, False Positive/Negative Rates
- **Visualizations:** Confusion matrix (with row percentages) + performance metrics bar chart

---

## 📝 Summary

> **SmartCity experiments confirmed a crucial insight:** perfect accuracy on synthetic data with hidden duplicates does not translate to real-world performance. After removing duplicates, the baseline dropped to ~52% accuracy, and temporal modeling (CNN + LSTM) failed to improve beyond 38% due to insufficient training data.
>
> These findings motivated the transition to the **Real-Life Violence Dataset (RLVD)** — a larger, more diverse, real-world corpus that better supports robust and generalizable violence detection models.

---

**Last Updated:** November 2025
