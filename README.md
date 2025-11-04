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

## 🧱 Models Trained

| Model | Description | Main Goal | Key Outcome |
|:-------|:-------------|:-----------|:-------------|
| **Baseline CNN (ResNet-18)** | Trained on single frames from SmartCity | Detect violence from appearance | 100% accuracy on SmartCity, but failed on real data (overfitting) |
| **Fine-Tuned CNN** | Baseline model fine-tuned on RWF-2000 | Domain adaptation | Improved generalization (61% accuracy on RWF-2000) |
| **CNN + LSTM** | ResNet-18 features + LSTM over 16-frame clips | Capture motion and temporal patterns | Slight improvement, but limited by small dataset (~400 clips) |

---

## 🧠 What We Learned

1. **SmartCity data is too limited and clean** — models can memorize the dataset easily but fail to generalize.
2. **Adding motion modeling (CNN + LSTM)** helped slightly, but not enough with only a few hundred short clips.
3. **Real-world datasets are essential** — lighting, crowd density, and camera angles in real CCTV differ significantly.

---

## 🗂 Next Step: Move to a Larger Dataset

We decided to move beyond SmartCity and use **larger, real-world datasets** for better model generalization.  
Since `.avi` video files were too large and inefficient, we found another dataset containing **~2,000 `.mp4` clips**, which is much lighter and easier to process.  

This new dataset will be the **main focus** moving forward — where we’ll apply transfer learning, motion-aware models (e.g., Bi-LSTM or 3D CNNs), and stronger data augmentation.

---

## 🧩 Summary
> Our experiments on SmartCity confirmed that simple CNNs can perfectly fit synthetic data but fail in real-world conditions.  
> We’re now transitioning to a larger `.mp4`-based dataset to build more robust and realistic violence detection models.

---

**Last Updated:** November 2025
