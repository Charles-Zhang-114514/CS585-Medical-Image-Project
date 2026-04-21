"""
Cross-domain analysis: compute all metrics for the four experimental settings
and dump results for downstream figures / paper tables.

Inputs (from outputs/predictions/):
    CheXpert val:
        chexpert_val_resnet_preds.npy     (probs, shape (202, 2))
        chexpert_val_resnet_labels.npy
        chexpert_val_densenet_preds.npy
        chexpert_val_densenet_labels.npy
    MIMIC test (cross-domain):
        mimic_resnet_logits.npy           shape (1835, 2)
        mimic_resnet_probs.npy            shape (1835, 2)
        mimic_densenet_logits.npy
        mimic_densenet_probs.npy
        mimic_labels.npy                  values in {0.0, 1.0, -1.0}
        mimic_subset_tags.npy             strings

Outputs (in outputs/analysis/):
    cross_domain_results.json    all metrics with bootstrap CIs
    temperature_values.json      fitted T per model

Settings (per model, per pathology):
    1. in_domain            — CheXpert val, uncalibrated
    2. cross_domain_all     — MIMIC all, uncalibrated
    3. cross_domain_ts      — MIMIC all, temperature-scaled
    4. cross_domain_shared  — MIMIC shared_only, uncalibrated
    5. cross_domain_extra   — MIMIC extra_pathology, uncalibrated

Settings 4 & 5 together answer proposal's Setting (4).

Usage (from Spyder):
    %runfile E:/CS585-Project/scripts/analyze_cross_domain.py --wdir
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(r"E:\CS585-Project")
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import (
    compute_ece,
    auc_with_ci,
    ece_with_ci,
    bootstrap_metric,
)
from sklearn.metrics import roc_auc_score

PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"
OUT_DIR  = PROJECT_ROOT / "outputs" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["Pneumothorax", "Pleural Effusion"]
MODELS = ["resnet", "densenet"]
N_BOOTSTRAP = 2000
SEED = 42


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def inv_sigmoid(p, eps=1e-7):
    """Reconstruct logits from sigmoid probabilities. Clamp to avoid inf."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_temperature(logits, labels, lr=0.01, max_iter=100):
    """
    Fit a single scalar temperature T by minimizing BCE loss on (logits, labels).
    Operates column-by-column on multi-label logits; returns one T per column.

    Args:
        logits: (N, C) numpy array of raw logits
        labels: (N, C) numpy array of 0/1 labels
    Returns:
        list of floats, length C — optimal T for each label column
    """
    n_cols = logits.shape[1]
    temperatures = []
    for c in range(n_cols):
        logits_t = torch.tensor(logits[:, c], dtype=torch.float32)
        labels_t = torch.tensor(labels[:, c], dtype=torch.float32)

        # T is parameterized as log(T) so optimizer search is unconstrained
        log_T = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            T = torch.exp(log_T)
            scaled = logits_t / T
            loss = F.binary_cross_entropy_with_logits(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        T_val = float(torch.exp(log_T).item())
        temperatures.append(T_val)
    return temperatures


def conf_on_incorrect(preds, labels, threshold=0.5):
    """Mean confidence on incorrect predictions. Confidence = max(p, 1-p)."""
    pred_cls = (preds >= threshold).astype(np.float32)
    incorrect = pred_cls != labels
    if incorrect.sum() == 0:
        return float("nan")
    confidence = np.maximum(preds, 1.0 - preds)
    return float(confidence[incorrect].mean())


def brier_score(preds, labels):
    return float(np.mean((preds - labels) ** 2))


# -----------------------------------------------------------------------------
# Metric bundle: compute all metrics with CI for one (preds, labels) pair
# -----------------------------------------------------------------------------
def compute_all_metrics(preds, labels, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Compute AUC, ECE, Brier, conf-on-incorrect, each with bootstrap CI.
    Returns a dict with nested {metric_name: {point, lower, upper, std}}.
    """
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)

    out = {
        "n_total":    int(len(labels)),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }

    # AUC needs both classes present; otherwise we report null
    if n_pos == 0 or n_neg == 0:
        out["auc"] = None
    else:
        out["auc"] = auc_with_ci(preds, labels, n_iter=n_bootstrap, seed=seed)

    out["ece"]   = ece_with_ci(preds, labels, n_iter=n_bootstrap, seed=seed)
    out["brier"] = bootstrap_metric(brier_score, preds, labels,
                                    n_iter=n_bootstrap, seed=seed)
    out["conf_on_incorrect"] = bootstrap_metric(
        conf_on_incorrect, preds, labels,
        n_iter=n_bootstrap, seed=seed,
    )
    return out


# -----------------------------------------------------------------------------
# Filter for valid labels (drop -1 uncertain samples per pathology)
# -----------------------------------------------------------------------------
def filter_valid(preds_col, labels_col):
    """Drop samples with label == -1.0. Returns (preds, labels) 1-D arrays."""
    mask = labels_col != -1.0
    return preds_col[mask], labels_col[mask]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Loading predictions")
    print("=" * 70)

    # ---- CheXpert val (in-domain) ----
    cxp_probs = {}
    for m in MODELS:
        cxp_probs[m] = np.load(PRED_DIR / f"chexpert_val_{m}_preds.npy")
    cxp_labels = np.load(PRED_DIR / f"chexpert_val_resnet_labels.npy")
    # (labels are identical across both models — asserted when dumped)
    print(f"CheXpert val:  shape {cxp_probs['resnet'].shape}")

    # ---- MIMIC (cross-domain) ----
    mimic_logits = {m: np.load(PRED_DIR / f"mimic_{m}_logits.npy") for m in MODELS}
    mimic_probs  = {m: np.load(PRED_DIR / f"mimic_{m}_probs.npy")  for m in MODELS}
    mimic_labels = np.load(PRED_DIR / "mimic_labels.npy")
    mimic_tags   = np.load(PRED_DIR / "mimic_subset_tags.npy", allow_pickle=True)
    print(f"MIMIC all:     shape {mimic_probs['resnet'].shape}")
    print(f"  subset split: shared_only={int((mimic_tags=='shared_only').sum())}, "
          f"extra_pathology={int((mimic_tags=='extra_pathology').sum())}")

    # ---- Fit temperature from CheXpert val (logits reconstructed from probs) ----
    print("\n" + "=" * 70)
    print("Fitting temperature from CheXpert val")
    print("=" * 70)
    temperatures = {}
    for m in MODELS:
        cxp_logits_m = inv_sigmoid(cxp_probs[m])
        Ts = fit_temperature(cxp_logits_m, cxp_labels)
        temperatures[m] = dict(zip(TARGET_COLS, Ts))
        print(f"  {m}:  " +
              "  ".join(f"{p}={t:.4f}" for p, t in temperatures[m].items()))

    with open(OUT_DIR / "temperature_values.json", "w") as f:
        json.dump(temperatures, f, indent=2)
    print(f"\nSaved: temperature_values.json")

    # ---- Apply TS to MIMIC logits ----
    mimic_probs_ts = {}
    for m in MODELS:
        logits = mimic_logits[m]
        Ts = np.array([temperatures[m][p] for p in TARGET_COLS])  # per-pathology T
        scaled = logits / Ts[None, :]
        mimic_probs_ts[m] = sigmoid(scaled)

    # -------------------------------------------------------------------------
    # Compute metrics across all settings
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Computing metrics (bootstrap N={}) ".format(N_BOOTSTRAP))
    print("=" * 70)

    results = {}  # {setting: {model: {pathology: {metric_bundle}}}}

    # Helper: for each (preds_matrix, labels_matrix, sample_mask), run metrics
    # per pathology column.
    def eval_one_setting(probs_m, labels_m, sample_mask=None):
        """Return {pathology: metric_bundle} for one (preds, labels) pair."""
        out = {}
        for col, pname in enumerate(TARGET_COLS):
            if sample_mask is not None:
                p_col = probs_m[sample_mask, col]
                l_col = labels_m[sample_mask, col]
            else:
                p_col = probs_m[:, col]
                l_col = labels_m[:, col]
            # Exclude -1 labels per pathology
            p_valid, l_valid = filter_valid(p_col, l_col)
            out[pname] = compute_all_metrics(p_valid, l_valid)
        return out

    # Setting 1: in-domain (CheXpert val, uncalibrated)
    print("\n[1/5] in_domain (CheXpert val)")
    results["in_domain"] = {}
    for m in MODELS:
        results["in_domain"][m] = eval_one_setting(cxp_probs[m], cxp_labels)

    # Setting 2: cross-domain, uncalibrated, all
    print("[2/5] cross_domain_all (MIMIC all, uncalibrated)")
    results["cross_domain_all"] = {}
    for m in MODELS:
        results["cross_domain_all"][m] = eval_one_setting(mimic_probs[m], mimic_labels)

    # Setting 3: cross-domain, TS, all
    print("[3/5] cross_domain_ts  (MIMIC all, temperature-scaled)")
    results["cross_domain_ts"] = {}
    for m in MODELS:
        results["cross_domain_ts"][m] = eval_one_setting(mimic_probs_ts[m], mimic_labels)

    # Setting 4: cross-domain, shared_only
    print("[4/5] cross_domain_shared  (MIMIC shared_only)")
    mask_shared = (mimic_tags == "shared_only")
    results["cross_domain_shared"] = {}
    for m in MODELS:
        results["cross_domain_shared"][m] = eval_one_setting(
            mimic_probs[m], mimic_labels, sample_mask=mask_shared)

    # Setting 5: cross-domain, extra_pathology
    print("[5/5] cross_domain_extra  (MIMIC extra_pathology)")
    mask_extra = (mimic_tags == "extra_pathology")
    results["cross_domain_extra"] = {}
    for m in MODELS:
        results["cross_domain_extra"][m] = eval_one_setting(
            mimic_probs[m], mimic_labels, sample_mask=mask_extra)

    # ---- Dump JSON ----
    results["_meta"] = {
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "temperatures": temperatures,
    }
    out_path = OUT_DIR / "cross_domain_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # -------------------------------------------------------------------------
    # Human-readable summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    setting_order = [
        ("in_domain",          "In-domain (CheXpert val)"),
        ("cross_domain_all",   "Cross-domain (MIMIC all)"),
        ("cross_domain_ts",    "Cross-domain + TS"),
        ("cross_domain_shared","Cross-domain (shared_only)"),
        ("cross_domain_extra", "Cross-domain (extra_pathology)"),
    ]

    for m in MODELS:
        print(f"\n--- {m.upper()} ---")
        header = f"{'Setting':<32} {'Pathology':<18} {'N':>5} {'pos':>4}  "
        header += f"{'AUC':>20}  {'ECE':>20}  {'conf-on-incorrect':>20}"
        print(header)
        print("-" * len(header))
        for key, label in setting_order:
            for pname in TARGET_COLS:
                r = results[key][m][pname]
                n     = r["n_total"]
                npos  = r["n_positive"]

                def fmt(d):
                    if d is None:
                        return "N/A".rjust(20)
                    return f"{d['point']:.3f} [{d['lower']:.3f},{d['upper']:.3f}]".rjust(20)

                auc_s  = fmt(r.get("auc"))
                ece_s  = fmt(r["ece"])
                coi_s  = fmt(r["conf_on_incorrect"])
                print(f"{label:<32} {pname:<18} {n:>5} {npos:>4}  "
                      f"{auc_s}  {ece_s}  {coi_s}")

    print("\nDone.")


if __name__ == "__main__":
    main()