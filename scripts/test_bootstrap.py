"""Sanity-check bootstrap CI on CheXpert val predictions."""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.eval.metrics import auc_with_ci, ece_with_ci  # noqa: E402

PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")

DISEASES = ["Pneumothorax", "Pleural Effusion"]
MODELS = {
    "ResNet-50": ("chexpert_val_resnet_preds.npy",
                  "chexpert_val_resnet_labels.npy"),
    "DenseNet-121": ("chexpert_val_densenet_preds.npy",
                     "chexpert_val_densenet_labels.npy"),
}

EXPECTED_AUC = {
    ("ResNet-50", "Pneumothorax"):      0.9136,
    ("ResNet-50", "Pleural Effusion"):  0.9322,
    ("DenseNet-121", "Pneumothorax"):   0.9348,
    ("DenseNet-121", "Pleural Effusion"): 0.9258,
}

SEP = "=" * 62


def main():
    for model_name, (pred_file, label_file) in MODELS.items():
        preds  = np.load(os.path.join(PRED_DIR, pred_file))
        labels = np.load(os.path.join(PRED_DIR, label_file))

        for col, disease in enumerate(DISEASES):
            p = preds[:, col]
            l = labels[:, col]
            n_pos = int(l.sum())

            print(SEP)
            print(f"  {model_name}  |  {disease}")
            print(f"  Samples: {len(l)}  (positive: {n_pos})")
            print(SEP)

            auc_res = auc_with_ci(p, l, n_iter=2000, seed=42)
            ece_res = ece_with_ci(p, l, n_iter=2000, seed=42)

            expected = EXPECTED_AUC.get((model_name, disease))
            auc_match = ""
            if expected is not None:
                ok = abs(auc_res["point"] - expected) < 0.001
                auc_match = f"  (expected {expected:.4f} — {'OK' if ok else 'MISMATCH'})"

            print(f"  AUC  : {auc_res['point']:.4f}  "
                  f"[{auc_res['lower']:.4f}, {auc_res['upper']:.4f}]  "
                  f"std={auc_res['std']:.4f}{auc_match}")
            print(f"  ECE  : {ece_res['point']:.4f}  "
                  f"[{ece_res['lower']:.4f}, {ece_res['upper']:.4f}]  "
                  f"std={ece_res['std']:.4f}")

            in_ci = auc_res["lower"] <= auc_res["point"] <= auc_res["upper"]
            print(f"  AUC point in CI: {in_ci}")
            in_ci = ece_res["lower"] <= ece_res["point"] <= ece_res["upper"]
            print(f"  ECE point in CI: {in_ci}")
            print()

    print(SEP)
    print("  Done. Wide CIs on Pneumothorax are expected (few positives).")
    print(SEP)


if __name__ == "__main__":
    main()
