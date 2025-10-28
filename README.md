# 🎥 Violence Detection Using CCTV (MSAI 349 – Group 9)

**Team Members:**  
- Chia-Lu Karl Ma (lro6877)  
- Jamal Moussa (xrz2515)  
- Ramakrishna Saravanan (cqp0132)

---

## 🧩 Project Overview

This project aims to detect **violent incidents in real-time CCTV footage** to support faster emergency response and public safety.  

We build and compare multiple models to classify violence from surveillance videos, progressing from simple **frame-based CNNs** to **temporal models** that analyze motion over time.

### Datasets Used
1. **SmartCity CCTV Violence Detection (SCVD)**  
   - Short, 1-second CCTV clips labeled as:  
     - `normal` — no violence  
     - `violence` — physical fights  
     - `weaponized` — fights involving handheld weapons  
   - Used for **training** and **within-dataset evaluation**

2. **RWF-2000 (Real-World Fights Dataset)**  
   - Real surveillance footage of public fights (fight vs non-fight)  
   - Used for **cross-dataset generalization testing** and fine-tuning

---

## 🧠 Current Progress

### ✅ Environment Setup
- Virtual environment created with **Python 3.12**
- Installed dependencies:
  ```bash
  pip install -r requirements.txt
  pip install torch torchvision torchaudio
  ```
- Verified GPU support (Apple MPS backend)

---

### ✅ Data Organization

Our repository now stores **both datasets** in a unified structure for clarity and flexibility.

```
data/
├── smart_city-processed_image/
│   ├── train_frames/
│   │   ├── normal/
│   │   ├── violence/
│   │   └── weaponized/
│   └── test_frames/
│       ├── normal/
│       ├── violence/
│       └── weaponized/
│
├── rwf2000-processed_image/
│   ├── train_frames/
│   │   ├── normal/        # NonFight
│   │   ├── violence/      # Fight
│   │   └── weaponized/    # (empty or optional)
│   └── test_frames/
│       ├── normal/
│       ├── violence/
│       └── weaponized/
│
└── combined/               # optional mixed-domain experiments (future)
```

---

### ✅ Data Processing Scripts
| Script | Purpose |
|---------|----------|
| `src/extract_frames.py` | Extracts the **middle frame** from each short clip (used for 2D CNN) |
| `src/check_duplicates.py` | Uses **perceptual hashing (pHash)** to verify that SCVD and RWF datasets have **no overlapping images** |
| `src/dataset_loader.py` | Loads frame datasets with consistent normalization and worker configuration |

---

### ✅ Model Development

#### **Step 3 – Baseline 2D CNN (ResNet-18)**
- Trained on SCVD middle-frame images  
- Achieved **~100% accuracy** on SCVD test data (overfitting)  
- Demonstrated perfect learning of within-dataset distribution

#### **Cross-Dataset Evaluation (SmartCity → RWF-2000)**
| Metric | SmartCity Test | RWF-2000 Test |
|:--------|:---------------|:--------------|
| Accuracy | 1.00 | 0.40 |
| Macro F1 | 1.00 | 0.36 |

**Interpretation:**  
The model performs perfectly on its own dataset but fails to generalize to RWF-2000 due to strong **domain shift** (camera angles, lighting, motion, and environment differences).  
This confirms **overfitting** and motivates **transfer learning** and **temporal modeling** as next steps.

---

### ✅ Evaluation Visualization
Each evaluation now includes a **confusion-matrix heatmap** with an overlaid metrics summary:

```
accuracy = 0.40
macro precision = 0.47
macro recall    = 0.43
macro f1-score  = 0.36
normal       F1 = 0.31
violence     F1 = 0.62
weaponized   F1 = 0.15
```

These plots are automatically generated in `/results/` for record keeping.

---

## 🧩 Next Steps

| Step | Focus | Description |
|------|--------|-------------|
| **3A. Fine-Tune Baseline** | 🧠 Transfer Learning | Load SmartCity weights and fine-tune last layers on RWF-2000. Expect performance recovery from 40% → ~80%. |
| **4. Temporal Modeling** | 🎥 Motion Awareness | Extend to CNN+LSTM or 3D CNN to capture temporal features across frames. |
| **5. Domain Fusion** | 🌍 Robustness | Train on mixed SmartCity + RWF data to test cross-domain generalization. |
| **6. Visualization** | 📊 Interpretability | Compare confusion matrices, per-class F1 trends, and frame-level vs video-level predictions. |

---

## 📊 Evaluation Metrics
- **Confusion Matrix** — visual misclassification summary  
- **Precision / Recall / F1-Score** — per-class balance  
- **Accuracy & Macro Averages** — overall model performance  
- **Per-Class F1 Display** — shown on graphs for clear comparison  

---

## 🧮 Repository Structure

```
msai349-violence-detection/
├── data/
│   ├── smart_city-processed_image/
│   └── rwf2000-processed_image/
├── models/
│   ├── baseline_cnn.pt
│   └── ...
├── src/
│   ├── extract_frames.py
│   ├── check_duplicates.py
│   ├── dataset_loader.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   └── utils.py
├── results/
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# 1️⃣ Activate environment
source venv/bin/activate

# 2️⃣ Extract frames from videos
python src/extract_frames.py

# 3️⃣ Train baseline CNN
python src/train_baseline.py

# 4️⃣ Evaluate on both datasets
python src/evaluate_baseline.py
```

---

## 🧭 Key Insights So Far
- 2D CNN learns dataset-specific patterns (overfitting).  
- Cross-dataset testing reveals poor generalization (40%).  
- RWF-2000 has no overlap with SCVD — datasets are clean.  
- Next focus: **fine-tuning** and **temporal learning** for real-world robustness.

---

**Last Updated:** October 2025
