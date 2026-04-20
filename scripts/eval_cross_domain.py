"""
Cross-domain evaluation: dump MIMIC-CXR predictions from CheXpert-trained models.

Runs inference on MIMIC-CXR test set (filtered: frontal + labeled + local) for
both ResNet-50 and DenseNet-121. Saves logits, sigmoid probabilities, labels,
and subset tags as .npy files for downstream analysis.

This script intentionally does NOT compute metrics — metric computation (AUC,
ECE, bootstrap CIs, temperature scaling, reliability diagrams) happens in a
separate analysis pass from these dumps. This keeps GPU inference a one-time
cost.

Output files (in outputs/predictions/):
    mimic_resnet_logits.npy      shape (N, 2), raw model outputs
    mimic_resnet_probs.npy       shape (N, 2), sigmoid(logits)
    mimic_densenet_logits.npy
    mimic_densenet_probs.npy
    mimic_labels.npy             shape (N, 2), values in {0.0, 1.0, -1.0}
    mimic_subset_tags.npy        shape (N,), strings

Column order for logits/probs/labels: [Pneumothorax, Pleural Effusion]

Usage (from Spyder):
    %runfile E:/CS585-Project/scripts/eval_cross_domain.py --wdir
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

from src.data.mimic_loader import MIMICCXRDataset, build_mimic_test_manifest
from src.models.classifiers import get_resnet50, get_densenet121

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIMIC_ROOT = PROJECT_ROOT / "data" / "raw" / "mimic"
CKPT_DIR   = PROJECT_ROOT / "outputs" / "checkpoints"
OUT_DIR    = PROJECT_ROOT / "outputs" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE  = 32
NUM_WORKERS = 0  # Windows + Spyder: safest at 0

# Val transform: identical to CheXpert val transform (no augmentation)
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
    """
    Run inference and return (logits, probs, labels, subset_tags).

    All numeric outputs are numpy arrays; subset_tags is a list of strings
    (cast to np.array of dtype object by caller).
    """
    model.eval()
    all_logits, all_labels, all_tags = [], [], []

    for images, labels, tags in loader:
        images = images.to(DEVICE)
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
        # `tags` is a tuple of strings of length batch_size (default collate)
        all_tags.extend(tags)

    logits = np.concatenate(all_logits)
    probs  = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    labels = np.concatenate(all_labels)
    tags   = np.array(all_tags, dtype=object)
    return logits, probs, labels, tags


def dump_one_model(model_name, model_fn, ckpt_name, loader):
    """Run inference for one model, save logits + probs, return labels/tags."""
    print(f"\n{'=' * 62}")
    print(f"Model: {model_name}")
    print(f"{'=' * 62}")

    # Build model + load weights
    model = model_fn(num_classes=2, pretrained=False).to(DEVICE)
    ckpt_path = CKPT_DIR / ckpt_name
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {ckpt_path.name}")

    # Inference
    logits, probs, labels, tags = run_inference(model, loader)
    print(f"Logits shape: {logits.shape}   "
          f"range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"Probs shape:  {probs.shape}    "
          f"range: [{probs.min():.4f}, {probs.max():.4f}]")

    # Save logits + probs (not labels/tags — those are shared, saved once)
    np.save(OUT_DIR / f"mimic_{model_name}_logits.npy", logits)
    np.save(OUT_DIR / f"mimic_{model_name}_probs.npy",  probs)
    print(f"Saved: mimic_{model_name}_logits.npy")
    print(f"Saved: mimic_{model_name}_probs.npy")

    return labels, tags


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Output dir: {OUT_DIR}")

    # Build manifest once (filesystem scan + label join happens here)
    manifest = build_mimic_test_manifest(MIMIC_ROOT, verbose=True)

    # Build dataset + loader (shared across both models)
    dataset = MIMICCXRDataset(
        MIMIC_ROOT,
        transform=val_tf,
        subset="all",
        manifest=manifest,
    )
    print(f"\nDataset: {dataset}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ---- Run inference for both models ----
    r_labels, r_tags = dump_one_model(
        "resnet", get_resnet50, "resnet50_best.pth", loader,
    )
    d_labels, d_tags = dump_one_model(
        "densenet", get_densenet121, "densenet121_best.pth", loader,
    )

    # Sanity: labels and tags must be identical across both models
    assert np.array_equal(r_labels, d_labels), "Labels mismatch across models!"
    assert np.array_equal(r_tags,   d_tags),   "Tags mismatch across models!"
    print("\n✓ Labels and subset_tags match across both models.")

    # Save shared labels + tags once
    np.save(OUT_DIR / "mimic_labels.npy",       r_labels)
    np.save(OUT_DIR / "mimic_subset_tags.npy",  r_tags)
    print(f"Saved: mimic_labels.npy       shape {r_labels.shape}")
    print(f"Saved: mimic_subset_tags.npy  shape {r_tags.shape}")

    # ---- Sanity summary ----
    print(f"\n{'=' * 62}")
    print("Sanity summary (label distribution)")
    print(f"{'=' * 62}")

    TARGET_COLS = ["Pneumothorax", "Pleural Effusion"]
    for i, col in enumerate(TARGET_COLS):
        pos        = int((r_labels[:, i] ==  1.0).sum())
        neg        = int((r_labels[:, i] ==  0.0).sum())
        uncertain  = int((r_labels[:, i] == -1.0).sum())
        print(f"  {col:20s}  pos={pos:>4}  neg={neg:>4}  "
              f"uncertain={uncertain:>3}  (total={len(r_labels)})")

    n_shared = int((r_tags == "shared_only").sum())
    n_extra  = int((r_tags == "extra_pathology").sum())
    print(f"\n  Subset tags: shared_only={n_shared}, extra_pathology={n_extra}")

    print(f"\n{'=' * 62}")
    print("Done. Files ready for cross-domain analysis:")
    print(f"{'=' * 62}")
    for f in sorted(OUT_DIR.glob("mimic_*.npy")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:35s}  ({size_kb:>6.1f} KB)")