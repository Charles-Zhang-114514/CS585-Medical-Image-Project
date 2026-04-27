import json
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Make src/ importable when run from terminal
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import compute_reliability_diagram_data


# Paths
# REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
PRED_DIR = REPO_ROOT / "outputs" / "predictions"

RESULTS_PATH = ANALYSIS_DIR / "cross_domain_results.json"

FIGURES_DIR=REPO_ROOT/"figures"


# Constants
SETTINGS = [
    "in_domain",
    "cross_domain_all",
    "cross_domain_ts",
    "cross_domain_shared",
    "cross_domain_extra",
]

# Two models being evaluated
MODELS = ["resnet", "densenet"]

# Two target pathologies (binary classification per column)
PATHOLOGIES = ["Pneumothorax", "Pleural Effusion"]

# Column index mapping for predictions/labels arrays
PATHOLOGY_TO_IDX = {
    "Pneumothorax": 0,
    "Pleural Effusion": 1,
}

# Number of bins used in reliability diagram
N_BINS = 15


# Helpers
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_ece_from_bins(bin_acc, bin_conf, bin_counts):
    total = bin_counts.sum()
    if total == 0:
        return 0.0

    mask = bin_counts > 0
    ece = np.sum(
        (bin_counts[mask] / total) *
        np.abs(bin_acc[mask] - bin_conf[mask])
    )
    return float(ece)


# Load all data
def load_data():
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    data = {
        "results": results,

        # MIMIC (cross-domain) probabilities and logits
        "mimic_resnet_probs": np.load(PRED_DIR / "mimic_resnet_probs.npy"),
        "mimic_densenet_probs": np.load(PRED_DIR / "mimic_densenet_probs.npy"),
        "mimic_resnet_logits": np.load(PRED_DIR / "mimic_resnet_logits.npy"),
        "mimic_densenet_logits": np.load(PRED_DIR / "mimic_densenet_logits.npy"),
        
        # MIMIC labels and subset tags
        "mimic_labels": np.load(PRED_DIR / "mimic_labels.npy", allow_pickle=True),
        "mimic_subset_tags": np.load(PRED_DIR / "mimic_subset_tags.npy", allow_pickle=True),

        # CheXpert validation (in-domain)
        "chexpert_val_resnet_preds": np.load(PRED_DIR / "chexpert_val_resnet_preds.npy"),
        "chexpert_val_densenet_preds": np.load(PRED_DIR / "chexpert_val_densenet_preds.npy"),
        "chexpert_val_resnet_labels": np.load(PRED_DIR / "chexpert_val_resnet_labels.npy", allow_pickle=True),
        "chexpert_val_densenet_labels": np.load(PRED_DIR / "chexpert_val_densenet_labels.npy", allow_pickle=True),
    }

    return data


# Get probs + labels per case
def get_probs_and_labels(data, setting, model, pathology):
    idx = PATHOLOGY_TO_IDX[pathology]

    if setting == "in_domain":
        # CheXpert validation data
        probs = data[f"chexpert_val_{model}_preds"][:, idx]
        labels = data[f"chexpert_val_{model}_labels"][:, idx]

    elif setting == "cross_domain_all":
        # Full MIMIC dataset (uncalibrated)
        probs = data[f"mimic_{model}_probs"][:, idx]
        labels = data["mimic_labels"][:, idx]

    elif setting == "cross_domain_ts":
        # Temperature scaling applied to logits
        logits = data[f"mimic_{model}_logits"][:, idx]
        labels = data["mimic_labels"][:, idx]

        # Apply T to logits BEFORE sigmoid
        T = data["results"]["_meta"]["temperatures"][model][pathology]
        probs = sigmoid(logits / T)

    elif setting == "cross_domain_shared":
        probs = data[f"mimic_{model}_probs"][:, idx]
        labels = data["mimic_labels"][:, idx]
        tags = data["mimic_subset_tags"]

        mask_subset = tags == "shared_only"
        mask_valid = labels != -1
        mask = mask_subset & mask_valid

        return probs[mask], labels[mask]

    elif setting == "cross_domain_extra":
        probs = data[f"mimic_{model}_probs"][:, idx]
        labels = data["mimic_labels"][:, idx]
        tags = data["mimic_subset_tags"]

        mask_subset = tags == "extra_pathology"
        mask_valid = labels != -1
        mask = mask_subset & mask_valid

        return probs[mask], labels[mask]

    else:
        raise ValueError(f"Unknown setting: {setting}")

    # remove invalid labels
    mask = labels != -1
    return probs[mask], labels[mask]




#figure 
def plot(d, p):
    r=d["results"]
    f, x=plt.subplots(2,5,figsize=(20,9))
    for i in range(len(MODELS)):
        for j in range(len(SETTINGS)):
            m=MODELS[i]
            s=SETTINGS[j]
        
            l=x[i,j]
            prob, y=get_probs_and_labels(d,s, m, p)
            c=compute_reliability_diagram_data(y, prob,n_bins=N_BINS)
            ma=c["bin_counts"]>0
            l.bar(c["bin_confidences"][ma],c["bin_accuracies"][ma],width=1.0/N_BINS,color="lightcoral")
            l.plot([0,1],[0,1],'--',color='black')
            l.grid(True,linestyle="--",alpha=0.5)
            e=r[s][m][p]["ece"]["point"]
            l.set_title(f"{m} — {s}\nECE={e:.3f}")
            l.set_xlim(0,1)
            l.set_ylim(0,1)
    plt.tight_layout()
    return f









# Main computation
def main():
    data = load_data()
    results = data["results"]

    print("\n=== VALIDATION START ===\n")

    all_ok = True

    # Loop over all 20 cases (5 settings x 2 models x 2 pathologies)
    for setting in SETTINGS:
        for model in MODELS:
            for pathology in PATHOLOGIES:

                probs, labels = get_probs_and_labels(
                    data, setting, model, pathology
                )

                # check count against JSON
                expected_n = results[setting][model][pathology]["n_total"]
                actual_n = len(labels)

                # compute reliability diagram bins
                bin_data = compute_reliability_diagram_data(
                    labels, probs, n_bins=N_BINS
                )

                bin_acc = bin_data["bin_accuracies"]
                bin_conf = bin_data["bin_confidences"]
                bin_counts = bin_data["bin_counts"]

                # compute ECE from bins
                ece_computed = compute_ece_from_bins(
                    bin_acc, bin_conf, bin_counts
                )

                ece_json = results[setting][model][pathology]["ece"]["point"]

                # check match
                match = np.isclose(ece_computed, ece_json, atol=1e-4)

                if not match:
                    all_ok = False

                # print validation summary
                print(f"{setting:20} | {model:8} | {pathology:18}")
                print(f"  n: {actual_n} (expected {expected_n})")
                print(f"  ECE: {ece_computed:.6f} (json {ece_json:.6f})")
                print(f"  MATCH: {match}")
                print()

    print("\n=== DONE ===")
    print("ALL MATCH:", all_ok)

    # save fig
    FIGURES_DIR.mkdir(parents=True,exist_ok=True)
    for p in PATHOLOGIES:
        fig=plot(data,p)
        name=p.lower().replace(" ","_")
        fig.savefig(FIGURES_DIR/ f"reliability_{name}.png",dpi=300)
        plt.close(fig)
    


if __name__ == "__main__":
    main()