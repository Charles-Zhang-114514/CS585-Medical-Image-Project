# CS585 Project — Chest X-ray Model Confidence Under Cross-Dataset Deployment

> **Reliability of Deep Learning Model Confidence in Medical Imaging**  
> Boston University · CS585 Image & Video Computing

---

## Overview

This project investigates whether deep learning model confidence remains reliable
when transferring from **CheXpert** to **MIMIC-CXR**, and whether **temperature scaling**
can improve calibration under dataset shift.

Deep learning models often achieve high performance on internal datasets but may behave
unpredictably when deployed on data from different hospitals. This project jointly analyzes
predictive performance and confidence calibration under cross-dataset deployment.

---

## Research Goals

- Analyze how model confidence behaves under **cross-dataset deployment**
- Evaluate whether simple calibration methods (e.g., **temperature scaling**) improve reliability
- Compare reliability behavior across diseases with different prevalence patterns
  (e.g., Pneumothorax vs. Pleural Effusion)

---

## Target Diseases

Experiments focus on two chest X-ray pathologies, chosen for their contrasting prevalence in CheXpert:

| Disease | Prevalence in CheXpert |
|---|---|
| **Pneumothorax** | Relatively rare |
| **Pleural Effusion** | Relatively common |

---

## Datasets

> ⚠️ Datasets are **not included** in this repository due to size and license restrictions.

### CheXpert (small version)
- Source: Stanford ML Group
- Kaggle mirror: https://www.kaggle.com/datasets/ashery/chexpert
- Size: approximately **10–11 GB**
- Role: **Training / validation / in-domain evaluation**

> This project uses **CheXpert-small** for computational efficiency while maintaining
> sufficient scale for evaluating model reliability and calibration.

### MIMIC-CXR
- Source: MIT + Beth Israel Deaconess Medical Center
- 377,110 images corresponding to 227,835 radiographic studies
- Version: **v2.1.0** (published July 23, 2024)
- Access: [PhysioNet](https://physionet.org/content/mimic-cxr/2.1.0/) — requires credentialed account + signed DUA
- Role: **Cross-dataset evaluation**

---

## Model

We train and evaluate two CNN-based classifiers:

| Model | Source |
|---|---|
| **ResNet-50** | torchvision |
| **DenseNet-121** | [TorchXRayVision](https://github.com/mlmed/torchxrayvision) |

Including two architectures allows us to examine whether confidence miscalibration
under domain shift is model-specific or consistent across different CNN designs.

Relevant pathology indices for DenseNet-121:

| Pathology | Index |
|---|---|
| Pneumothorax | 3 |
| Effusion | 7 |

---

## Experimental Pipeline

```
1️⃣  Train ResNet-50 & DenseNet-121 on CheXpert
2️⃣  Evaluate in-domain performance on CheXpert test set
3️⃣  Evaluate cross-domain performance on MIMIC-CXR test set
4️⃣  Measure reliability metrics:
        • AUC
        • Expected Calibration Error (ECE)
        • Brier Score
        • Confidence on incorrect predictions
5️⃣  Apply Temperature Scaling (learned on CheXpert validation set)
6️⃣  Re-evaluate across four settings:
        (1) Uncalibrated model — in-domain
        (2) Uncalibrated model — cross-domain
        (3) Temperature-scaled model — cross-domain
        (4) Uncalibrated model — stronger OOD samples in MIMIC-CXR
7️⃣  Grad-CAM visualization on high-confidence failure cases
```

---

## Project Structure

```
CS585-Project/
├── src/
│   ├── data/              # Dataset loaders
│   ├── models/            # Model definitions
│   ├── eval/              # Evaluation metrics
│   ├── calibration/       # Calibration methods
│   └── interpretability/  # Grad-CAM tools
├── scripts/               # Environment checks, dataset inspection, test scripts
├── data/
│   ├── raw/               # Local datasets (not tracked by git)
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── logs/
│   └── checkpoints/
└── notebooks/
```

---

## Current Progress

**Completed**
- [x] Environment setup
- [x] TorchXRayVision inference test
- [x] CheXpert metadata inspection
- [x] Project repository initialization

**Next Steps**
- [ ] Implement PyTorch Dataset for CheXpert & MIMIC-CXR
- [ ] Build DataLoader pipeline
- [ ] Train ResNet-50 and DenseNet-121 on CheXpert
- [ ] Implement ECE, Brier Score, and calibration metrics
- [ ] Apply temperature scaling
- [ ] Cross-dataset evaluation on MIMIC-CXR
- [ ] Grad-CAM visualization of failure cases

---

## Requirements

```bash
pip install torch torchvision torchxrayvision numpy pandas scikit-learn matplotlib
```

---

## Authors

Boston University · CS585 Image & Video Computing
