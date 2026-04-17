"""
Dump model predictions on CheXpert validation set.

Creates numpy files containing (preds, labels) pairs for downstream analysis
(bootstrap CI, calibration analysis, etc.).

Output files go to outputs/predictions/:
    chexpert_val_resnet_preds.npy    shape (N, 2), sigmoid probabilities
    chexpert_val_resnet_labels.npy   shape (N, 2), binary 0/1
    chexpert_val_densenet_preds.npy
    chexpert_val_densenet_labels.npy

Column order (for both preds and labels): [Pneumothorax, Pleural Effusion]

Usage (from Spyder):
    %runfile E:/CS585-Project/scripts/dump_predictions.py --wdir
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Make src/ importable
PROJECT_ROOT = Path(r"E:\CS585-Project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chexpert_loader import CheXpertDataset
from src.models.classifiers import get_resnet50, get_densenet121

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "chexpert"
VAL_CSV   = DATA_ROOT / "valid.csv"
CKPT_DIR  = PROJECT_ROOT / "outputs" / "checkpoints"
OUT_DIR   = PROJECT_ROOT / "outputs" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE  = 32
NUM_WORKERS = 0  # Windows + Spyder: safest at 0

# Val transform, exactly as used during training
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, loader):
    """Returns (probs, labels) as numpy arrays, shape (N, 2)."""
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def dump_one_model(model_name, model_fn, ckpt_name):
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")

    # Build model with pretrained=False (we'll load trained weights)
    model = model_fn(num_classes=2, pretrained=False).to(DEVICE)

    # Load checkpoint (raw state_dict, based on train.py)
    ckpt_path = CKPT_DIR / ckpt_name
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {ckpt_path.name}")

    # Dataset + loader
    dataset = CheXpertDataset(
        csv_path=str(VAL_CSV),
        data_root=str(DATA_ROOT),
        transform=val_tf,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)
    print(f"Dataset size: {len(dataset)}")

    # Inference
    probs, labels = run_inference(model, loader)
    print(f"Predictions shape: {probs.shape}   "
          f"(prob range: [{probs.min():.4f}, {probs.max():.4f}])")
    print(f"Labels shape:      {labels.shape}")

    # Sanity: positive counts
    for i, col in enumerate(CheXpertDataset.TARGET_COLS):
        pos = int(labels[:, i].sum())
        print(f"  {col:20s}  positives: {pos} / {len(labels)}")

    # Sanity: reproduce known AUC (should match training log ResNet=0.9229 / DenseNet=0.9303)
    from sklearn.metrics import roc_auc_score
    auc_p  = roc_auc_score(labels[:, 0], probs[:, 0])
    auc_pe = roc_auc_score(labels[:, 1], probs[:, 1])
    print(f"  Sanity AUC:  Pneumothorax={auc_p:.4f}  "
          f"Pleural Eff={auc_pe:.4f}  Mean={(auc_p+auc_pe)/2:.4f}")

    # Save
    preds_path  = OUT_DIR / f"chexpert_val_{model_name}_preds.npy"
    labels_path = OUT_DIR / f"chexpert_val_{model_name}_labels.npy"
    np.save(preds_path, probs)
    np.save(labels_path, labels)
    print(f"Saved: {preds_path.name}")
    print(f"Saved: {labels_path.name}")

    return probs, labels


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Label column order: {CheXpertDataset.TARGET_COLS}")

    r_probs, r_labels = dump_one_model("resnet",   get_resnet50,    "resnet50_best.pth")
    d_probs, d_labels = dump_one_model("densenet", get_densenet121, "densenet121_best.pth")

    # Sanity: labels should be identical across both models (same dataset)
    assert np.array_equal(r_labels, d_labels), "Labels mismatch across models!"
    print("\n✓ Label arrays match across both models.")

    print("\nDone. Files ready for Nuo & Mingyang:")
    for f in sorted(OUT_DIR.glob("chexpert_val_*.npy")):
        print(f"  {f}")