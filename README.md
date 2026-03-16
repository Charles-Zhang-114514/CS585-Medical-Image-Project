CS585 Project
Reliability of Chest X-ray Model Confidence under Cross-Dataset Deployment

This project studies the reliability of deep learning model confidence in medical imaging when deployed across datasets.

In particular, we evaluate whether model confidence remains reliable when transferring from CheXpert to MIMIC-CXR, and whether temperature scaling can improve calibration under dataset shift.

Research Goal

Deep learning models often achieve high performance on internal datasets but may behave unpredictably when deployed on data from different hospitals.

This project investigates:

How model confidence behaves under cross-dataset deployment

Whether simple calibration methods such as temperature scaling improve reliability

Target Diseases

The experiments focus on two chest X-ray pathologies:

Pneumothorax

Pleural Effusion

These were chosen because they represent different prevalence patterns in CheXpert:

Pneumothorax → relatively rare

Pleural Effusion → relatively common

Datasets

This project uses two public chest X-ray datasets.

CheXpert

Stanford ML Group
224,316 chest radiographs from 65,240 patients.

Used for:

training / in-domain evaluation
MIMIC-CXR

MIT + Beth Israel Deaconess Medical Center
Large-scale chest X-ray dataset.

Used for:

cross-dataset evaluation

Note:
Datasets are not included in this repository due to size and license restrictions.

Planned Pipeline

The experimental pipeline consists of the following stages:

1️⃣ Train or run pretrained model on CheXpert

2️⃣ Evaluate performance on CheXpert (in-domain)

3️⃣ Evaluate performance on MIMIC-CXR (cross-dataset)

4️⃣ Measure reliability metrics

AUC

Expected Calibration Error (ECE)

Brier Score

Confidence on incorrect predictions

5️⃣ Apply Temperature Scaling

6️⃣ Re-evaluate calibration and reliability

Model

Baseline model:

DenseNet121 (TorchXRayVision)

The model outputs predictions for 18 chest pathologies.

Relevant indices:

Pneumothorax → index 3
Effusion → index 7
Project Structure
CS585-Project/

src/
    data/              dataset loaders
    models/            model definitions
    eval/              evaluation metrics
    calibration/       calibration methods
    interpretability/  Grad-CAM tools

scripts/
    environment checks
    dataset inspection
    test scripts

data/
    raw/               local datasets (not tracked by git)
    processed/

outputs/
    figures/
    logs/
    checkpoints/

notebooks/
Current Progress

Completed:

Environment setup

TorchXRayVision inference test

CheXpert metadata inspection

Project repository initialization

Next steps:

Implement PyTorch Dataset for CheXpert

Build DataLoader pipeline

Run batch inference

Implement calibration metrics

Cross-dataset evaluation

Requirements

Main libraries:

torch
torchvision
torchxrayvision
numpy
pandas
scikit-learn
matplotlib
Authors

Boston University
CS585 Project# CS585-Project