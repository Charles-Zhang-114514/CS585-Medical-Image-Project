"""
Grad-CAM analysis for cross-domain high-confidence failures.

Visualizes where a CheXpert-trained model attends when it confidently
mispredicts Pneumothorax on MIMIC-CXR, compared to confidently correct
predictions.

Buckets (per model):
    FP (false positive): prob ≥ HIGH_CONF  AND label == 0
    FN (false negative): prob ≤ LOW_CONF   AND label == 1
    TP (true positive):  prob ≥ HIGH_CONF  AND label == 1

Outputs (in outputs/figures/gradcam/):
    gradcam_pneumo_<model>_fp.png
    gradcam_pneumo_<model>_fn.png
    gradcam_pneumo_<model>_tp.png

Usage (Spyder):
    %runfile E:/CS585-Project/scripts/gradcam_analysis.py --args "--model densenet"
    %runfile E:/CS585-Project/scripts/gradcam_analysis.py --args "--model resnet"

Usage (terminal):
    python scripts/gradcam_analysis.py --model densenet
    python scripts/gradcam_analysis.py --model resnet
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

PROJECT_ROOT = Path(r"E:\CS585-Project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mimic_loader import MIMICCXRDataset, build_mimic_test_manifest
from src.models.classifiers import get_resnet50, get_densenet121

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIMIC_ROOT = PROJECT_ROOT / "data" / "raw" / "mimic"
CKPT_DIR   = PROJECT_ROOT / "outputs" / "checkpoints"
PRED_DIR   = PROJECT_ROOT / "outputs" / "predictions"
OUT_DIR    = PROJECT_ROOT / "outputs" / "figures" / "gradcam"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATHOLOGY_NAME = "Pneumothorax"
PATHOLOGY_IDX  = 0

HIGH_CONF = 0.9
LOW_CONF  = 0.1
N_PER_BUCKET = 24

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

IM_MEAN = np.array([0.485, 0.456, 0.406])
IM_STD  = np.array([0.229, 0.224, 0.225])


# -----------------------------------------------------------------------------
# Model registry: one entry per supported model
# -----------------------------------------------------------------------------
# Each entry provides:
#   - ckpt_name: checkpoint filename in outputs/checkpoints/
#   - builder:   function returning an uninitialized model
#   - target_layer_fn: given a built model, return the layer to attach Grad-CAM
#   - pred_prefix: stem for the saved prediction .npy files
MODEL_REGISTRY = {
    "resnet": {
        "ckpt_name":        "resnet50_best.pth",
        "builder":          lambda: get_resnet50(num_classes=2, pretrained=False),
        "target_layer_fn":  lambda m: m.layer4,
        "pred_prefix":      "mimic_resnet",
        "display_name":     "ResNet-50",
    },
    "densenet": {
        "ckpt_name":        "densenet121_best.pth",
        "builder":          lambda: get_densenet121(num_classes=2, pretrained=False),
        "target_layer_fn":  lambda m: m.features.denseblock4,
        "pred_prefix":      "mimic_densenet",
        "display_name":     "DenseNet-121",
    },
}


# -----------------------------------------------------------------------------
# Phase 1: Bucket scan (no heatmaps yet)
# -----------------------------------------------------------------------------
def scan_buckets(cfg):
    probs  = np.load(PRED_DIR / f"{cfg['pred_prefix']}_probs.npy")[:, PATHOLOGY_IDX]
    labels = np.load(PRED_DIR / "mimic_labels.npy")[:, PATHOLOGY_IDX]

    valid = labels != -1.0
    orig_idx = np.arange(len(labels))[valid]
    probs, labels = probs[valid], labels[valid]

    fp_mask = (probs >= HIGH_CONF) & (labels == 0.0)
    fn_mask = (probs <= LOW_CONF)  & (labels == 1.0)
    tp_mask = (probs >= HIGH_CONF) & (labels == 1.0)

    buckets = {
        "fp": {"indices": orig_idx[fp_mask], "probs": probs[fp_mask]},
        "fn": {"indices": orig_idx[fn_mask], "probs": probs[fn_mask]},
        "tp": {"indices": orig_idx[tp_mask], "probs": probs[tp_mask]},
    }

    print(f"\n{'=' * 60}")
    print(f"Phase 1 — Bucket Scan ({PATHOLOGY_NAME}, {cfg['display_name']})")
    print(f"Total valid (non-uncertain) samples: {int(valid.sum())}")
    print(f"{'=' * 60}")
    desc = {
        "fp": f"False Positive (prob ≥ {HIGH_CONF}, label = 0)",
        "fn": f"False Negative (prob ≤ {LOW_CONF}, label = 1)",
        "tp": f"True Positive  (prob ≥ {HIGH_CONF}, label = 1)",
    }
    for key, b in buckets.items():
        print(f"  {desc[key]:50s} N = {len(b['indices'])}")
    return buckets


# -----------------------------------------------------------------------------
# Phase 2: Grad-CAM generation
# -----------------------------------------------------------------------------
def load_model(cfg):
    model = cfg["builder"]().to(DEVICE)
    ckpt_path = CKPT_DIR / cfg["ckpt_name"]
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path.name}")
    return model


def get_dataset():
    manifest = build_mimic_test_manifest(MIMIC_ROOT)
    return MIMICCXRDataset(
        MIMIC_ROOT, transform=val_tf, subset="all", manifest=manifest,
    )


def tensor_to_rgb(image_tensor):
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IM_STD + IM_MEAN
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def run_gradcam(model, target_layer, dataset, indices, probs_all, labels_all):
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(PATHOLOGY_IDX)]

    results = []
    for idx in indices:
        image, _, _ = dataset[idx]
        img_rgb = tensor_to_rgb(image)

        input_tensor = image.unsqueeze(0).to(DEVICE)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        overlay = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
        overlay = overlay.astype(np.float32) / 255.0

        results.append({
            "raw":     img_rgb,
            "overlay": overlay,
            "prob":    float(probs_all[idx]),
            "label":   int(labels_all[idx]),
        })
    return results


def plot_grid(results, title, save_path):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.5))
    if n == 1:
        axes = axes[:, None]

    for i, r in enumerate(results):
        axes[0, i].imshow(r["raw"])
        axes[0, i].axis("off")
        axes[0, i].set_title(f"p={r['prob']:.3f}  y={r['label']}", fontsize=10)
        axes[1, i].imshow(r["overlay"])
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Raw", fontsize=12)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=12)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(model_key):
    assert model_key in MODEL_REGISTRY, (
        f"Unknown model '{model_key}'. Choose from {list(MODEL_REGISTRY.keys())}"
    )
    cfg = MODEL_REGISTRY[model_key]

    # ---- Phase 1 ----
    buckets = scan_buckets(cfg)
    n_plan = {k: min(N_PER_BUCKET, len(b["indices"])) for k, b in buckets.items()}
    print(f"\nWill visualize: FP={n_plan['fp']}, FN={n_plan['fn']}, TP={n_plan['tp']}")

    if any(v == 0 for v in n_plan.values()):
        print("WARNING: At least one bucket is empty. Consider loosening "
              "HIGH_CONF / LOW_CONF thresholds.")

    # ---- Phase 2 ----
    model = load_model(cfg)
    target_layer = cfg["target_layer_fn"](model)
    dataset = get_dataset()

    probs_all  = np.load(PRED_DIR / f"{cfg['pred_prefix']}_probs.npy")[:, PATHOLOGY_IDX]
    labels_all = np.load(PRED_DIR / "mimic_labels.npy")[:, PATHOLOGY_IDX]

    print(f"\n{'=' * 60}")
    print(f"Phase 2 — Grad-CAM generation ({cfg['display_name']})")
    print(f"{'=' * 60}")

    bucket_configs = [
        ("fp",
         f"High-Confidence False Positives ({cfg['display_name']}, prob ≥ {HIGH_CONF}, label = 0)",
         f"gradcam_pneumo_{model_key}_fp.png", "desc"),
        ("fn",
         f"High-Confidence False Negatives ({cfg['display_name']}, prob ≤ {LOW_CONF}, label = 1)",
         f"gradcam_pneumo_{model_key}_fn.png", "asc"),
        ("tp",
         f"High-Confidence Correct Predictions ({cfg['display_name']}, prob ≥ {HIGH_CONF}, label = 1)",
         f"gradcam_pneumo_{model_key}_tp.png", "desc"),
    ]

    for key, title, fname, sort_order in bucket_configs:
        b = buckets[key]
        n = n_plan[key]
        if n == 0:
            print(f"\n[Skipping {key.upper()}: 0 samples]")
            continue

        sort_idx = np.argsort(b["probs"])
        if sort_order == "desc":
            sort_idx = sort_idx[::-1]
        selected = b["indices"][sort_idx][:n]

        print(f"\n[{key.upper()}] Generating Grad-CAM for {n} samples...")
        results = run_gradcam(model, target_layer, dataset, selected,
                              probs_all, labels_all)
        plot_grid(results, title, OUT_DIR / fname)

    print(f"\n{'=' * 60}")
    print(f"Done. {cfg['display_name']} figures in: {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="densenet",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Which model to analyze")
    args = parser.parse_args()
    main(args.model)