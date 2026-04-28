# CS585 Project — Chest X-ray Model Confidence Under Cross-Dataset Deployment

> **When Should We Trust Medical Image Models?**  
> Evaluating Confidence Reliability under Cross-Dataset Shift  
> Boston University · CS585 Image & Video Computing · Spring 2026
> 
> Instructor: Andrew Wood
---

## Overview

This project investigates whether deep learning model confidence remains reliable
when transferring from **CheXpert** (Stanford) to **MIMIC-CXR** (Beth Israel Deaconess),
and whether **temperature scaling** — a standard calibration method — improves reliability
across hospital systems.

Beyond predictive performance, we systematically analyze **calibration behavior** and use
**Grad-CAM** to reveal the mechanism behind cross-domain reliability gaps.

---

## Key Findings

1. **Cross-domain AUC degradation + rank reversal.** AUC drops by 7–13 percentage points 
   from CheXpert to MIMIC. The best in-domain model (DenseNet-121) is *not* the most 
   robust cross-domain model — ResNet-50 outperforms DenseNet under domain shift, 
   especially on Pneumothorax.

2. **Temperature scaling does not transfer safely.** While TS slightly improves overall 
   ECE in some settings, it **systematically increases confidence on incorrect predictions** 
   in **4/4** model-pathology combinations (with non-overlapping bootstrap CIs). 
   Source-domain calibration trades aggregate metric improvement for safety-critical regression.

3. **Mechanism explained via Grad-CAM.** DenseNet-121 relies on a strong spatial prior 
   (concentrated attention on the upper right lung field, reflecting CheXpert's 
   iatrogenic pneumothorax label skew). On false positives, the same attention pattern 
   appears even without pneumothorax. On false negatives, attention drifts to image 
   edges — a spurious shortcut. ResNet-50 shows distributed attention with no strong 
   spatial bias, explaining its better cross-domain robustness.

---

## Research Setup

### Target Pathologies

| Disease | Prevalence in CheXpert | Role |
|---|---|---|
| **Pneumothorax** | Relatively rare (~3.5%) | Class-imbalance stress test |
| **Pleural Effusion** | Relatively common (~32%) | Standard binary case |

### Datasets

> ⚠️ Datasets are **not included** in this repository due to size and license restrictions.

| Dataset | Source | Role |
|---|---|---|
| **CheXpert-small** | Stanford ML Group ([Kaggle](https://www.kaggle.com/datasets/ashery/chexpert)) | Training + in-domain evaluation |
| **MIMIC-CXR-JPG v2.1.0** | MIT / BIDMC ([PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.1.0/), credentialed access) | Cross-domain evaluation |

After filtering (test split → local existence → frontal view → labeled studies), 
the MIMIC evaluation set contains **1,835 samples**, with subset tags for downstream analysis:
- `shared_only`: 149 samples (only target pathologies present)
- `extra_pathology`: 1,686 samples (with co-occurring pathologies)

### Models

| Model | Pretrain | Use |
|---|---|---|
| **ResNet-50** | ImageNet (torchvision) | CheXpert fine-tuning, cross-domain eval |
| **DenseNet-121** | ImageNet (torchvision) | CheXpert fine-tuning, cross-domain eval |

Both models use 2-output heads for the two target pathologies. Identical training 
conditions ensure architectural fairness in the cross-domain comparison.

**Training:** Adam, lr=1e-4, batch size 32, 10 epochs, BCEWithLogitsLoss, ImageNet pretrained.  
**Best baselines:** ResNet-50 Mean AUC **0.9229** (epoch 4); DenseNet-121 **0.9303** (epoch 5).

---

## Experimental Pipeline

The pipeline follows a **"dump-once, analyze-many"** design: GPU inference runs once, 
all downstream analysis (bootstrap CIs, temperature scaling, subset filtering, Grad-CAM 
candidate selection) operates on stored `.npy` artifacts.

```
1. Train ResNet-50 & DenseNet-121 on CheXpert  →  scripts/train.py
2. Dump CheXpert val predictions               →  scripts/dump_predictions.py
3. Verify MIMIC subset feasibility             →  scripts/check_mimic_feasibility.py
4. Cross-domain inference                      →  scripts/eval_cross_domain.py
5. Compute metrics with bootstrap CIs          →  scripts/analyze_cross_domain.py
       Settings: in_domain, cross_domain_all, cross_domain_ts,
                 cross_domain_shared, cross_domain_extra
       Metrics: AUC, ECE, Brier, conf-on-incorrect (all with 95% CI)
6. Reliability diagrams                        →  scripts/plot_reliability_diagrams.py
7. Grad-CAM mechanism analysis                 →  scripts/gradcam_analysis.py
8. LaTeX tables for paper                      →  scripts/generate_latex_tables.py
```
---

## Project Structure

```
CS585-Project/
├── src/
│   ├── data/                         # Dataset loaders
│   │   ├── chexpert_loader.py        # CheXpertDataset, U-Ones policy
│   │   └── mimic_loader.py           # MIMICCXRDataset, subset tagging
│   ├── models/
│   │   └── classifiers.py            # ResNet-50 / DenseNet-121 factories
│   └── eval/
│       └── metrics.py                # ECE, Brier, conf-on-incorrect, bootstrap CI
├── scripts/                          # All experiment + analysis scripts
├── outputs/
│   ├── analysis/                     # cross_domain_results.json, temperature_values.json
│   ├── predictions/                  # Dumped logits/probs/labels (.npy)
│   ├── figures/
│   │   └── gradcam/                  # Grad-CAM visualizations
│   ├── tables/                       # auc_table.tex, calibration_table.tex
│   └── checkpoints/                  # Trained model weights (gitignored)
├── figures/                          # Reliability diagrams
├── presentation/                     # Slide deck (PDF + Google Slides link)
└── data/                             # Raw datasets (gitignored)
```

---

## Reproducing the Results

```bash
# 1. Train baselines (or place pretrained checkpoints in outputs/checkpoints/)
python scripts/train.py

# 2. Dump CheXpert val predictions
python scripts/dump_predictions.py

# 3. Run cross-domain inference (requires MIMIC-CXR data)
python scripts/eval_cross_domain.py

# 4. Compute all metrics with bootstrap CIs
python scripts/analyze_cross_domain.py

# 5. Generate reliability diagrams
python scripts/plot_reliability_diagrams.py

# 6. Generate Grad-CAM visualizations (run for each model)
python scripts/gradcam_analysis.py --model densenet
python scripts/gradcam_analysis.py --model resnet

# 7. Generate LaTeX tables
python scripts/generate_latex_tables.py
```

---

## Requirements

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib pillow grad-cam
```

---

## Team Contributions

**Haoran Zhang** (lead) — `zhr114@bu.edu`
- Research design and project leadership; author of proposal v1 and v2
- Dataset implementations (`CheXpertDataset`, `MIMICCXRDataset`); 
  cross-domain feasibility validation
- Baseline training (ResNet-50, DenseNet-121); MIMIC-CXR credentialed data access
- Cross-domain evaluation pipeline; per-pathology temperature scaling; 
  `analyze_cross_domain.py`
- Grad-CAM mechanism analysis; identification of DenseNet's spatial prior shortcut
- Team coordination and code review

**Sarah Lam** and **Yuting Lin** — `sarahl@bu.edu`, `linyt@bu.edu`

* CheXpert exploratory data analysis
* Training curve visualizations (loss + AUC vs. epoch)
* Reliability diagrams: data validation (matched against `cross_domain_results.json`) and plotting (4 settings × 2 models per pathology)

**Nuo Chen** and **Mingyang Li** — `ceno@bu.edu`, `limingy@bu.edu`

* Core evaluation metrics (`metrics.py`): ECE, Brier, confidence-on-incorrect, reliability diagram bins
* Bootstrap confidence interval extension (`bootstrap_metric`, `auc_with_ci`, `ece_with_ci`) with edge-case handling
* LaTeX results tables generation

---

## Presentation

Slide deck and supporting materials are available in [`presentation/`](presentation/).

---

## Acknowledgments

Boston University · CS585 Image and Video Computing · Spring 2026
