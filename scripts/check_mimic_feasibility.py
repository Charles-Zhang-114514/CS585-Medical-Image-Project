"""
Feasibility check for MIMIC-CXR cross-domain evaluation.

Verifies that after applying all filters (local JPG existence, test split,
frontal-only, exclude -1 labels), we still have enough samples in each
subset for the four experimental settings in the proposal.

Usage (from Spyder):
    %runfile E:/CS585-Project/scripts/check_mimic_feasibility.py --wdir
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MIMIC_DIR   = Path(r"E:\CS585-Project\data\raw\mimic")
IMAGES_DIR  = MIMIC_DIR / "images"

SPLIT_CSV    = MIMIC_DIR / "mimic-cxr-2.0.0-split.csv"
LABELS_CSV   = MIMIC_DIR / "mimic-cxr-2.0.0-chexpert.csv"
METADATA_CSV = MIMIC_DIR / "mimic-cxr-2.0.0-metadata.csv"

TARGET_PATHOLOGIES = ["Pneumothorax", "Pleural Effusion"]

# All 14 CheXpert-labeled columns; "extra" = everything except our 2 targets.
ALL_PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]
EXTRA_PATHOLOGIES = [p for p in ALL_PATHOLOGIES if p not in TARGET_PATHOLOGIES]

MIN_POSITIVES_FOR_AUC = 30  # below this, AUC is statistically unreliable


# -----------------------------------------------------------------------------
# Step 1: Load the three CSVs
# -----------------------------------------------------------------------------
print("=" * 70)
print("Loading CSVs...")
print("=" * 70)

splits   = pd.read_csv(SPLIT_CSV)
labels   = pd.read_csv(LABELS_CSV)
metadata = pd.read_csv(METADATA_CSV)

print(f"split.csv:    {len(splits):>8} rows  (dicom-level)")
print(f"chexpert.csv: {len(labels):>8} rows  (study-level)")
print(f"metadata.csv: {len(metadata):>8} rows  (dicom-level)")


# -----------------------------------------------------------------------------
# Step 2: Filter to test split
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 2: Filter to test split")
print("=" * 70)

test_split = splits[splits["split"] == "test"].copy()
print(f"Test dicoms (from split.csv): {len(test_split)}")


# -----------------------------------------------------------------------------
# Step 3: Scan local filesystem for actually-downloaded JPGs
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 3: Scan local filesystem")
print("=" * 70)

local_jpgs = list(IMAGES_DIR.rglob("*.jpg"))
print(f"Local JPGs found: {len(local_jpgs)}")

# Extract dicom_id from filename stem (filename is <dicom_id>.jpg)
local_dicom_ids = {p.stem for p in local_jpgs}
print(f"Unique dicom_ids locally: {len(local_dicom_ids)}")


# -----------------------------------------------------------------------------
# Step 4: Intersection of test split and local files
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 4: Test split ∩ Local files")
print("=" * 70)

test_local = test_split[test_split["dicom_id"].isin(local_dicom_ids)].copy()
print(f"Test dicoms present locally: {len(test_local)}  "
      f"(missing: {len(test_split) - len(test_local)})")


# -----------------------------------------------------------------------------
# Step 5: Join metadata to get ViewPosition, then filter frontal
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 5: Frontal filter (ViewPosition in {'PA','AP'})")
print("=" * 70)

# Only need dicom_id + ViewPosition from metadata
meta_subset = metadata[["dicom_id", "ViewPosition"]]
test_local_meta = test_local.merge(meta_subset, on="dicom_id", how="left")

print("ViewPosition distribution in test∩local:")
print(test_local_meta["ViewPosition"].value_counts(dropna=False))

frontal = test_local_meta[test_local_meta["ViewPosition"].isin(["PA", "AP"])].copy()
print(f"\nFrontal (PA or AP) dicoms: {len(frontal)}")


# -----------------------------------------------------------------------------
# Step 6: Join labels (study-level) onto frontal dicoms
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 6: Attach chexpert labels (join on study_id)")
print("=" * 70)

# Join on study_id (and subject_id to be safe).
df = frontal.merge(
    labels,
    on=["subject_id", "study_id"],
    how="left",
)

# Check how many got matched labels
unmatched = df[df[TARGET_PATHOLOGIES[0]].isna() & df[TARGET_PATHOLOGIES[1]].isna()]
print(f"Frontal dicoms with no matching label row: {len(unmatched)}")
print(f"Frontal dicoms with matched labels: {len(df) - len(unmatched)}")

# For the rest of the analysis, keep only matched
df = df[~df.index.isin(unmatched.index)].copy()


# -----------------------------------------------------------------------------
# Step 7: Per-pathology label distribution under U-zeros evaluation policy
# -----------------------------------------------------------------------------
# Evaluation policy: blank (NaN) -> negative (0), -1 -> EXCLUDE (set to NaN),
# 1.0 -> positive, 0.0 -> negative.
#
# For AUC, we drop rows where the target label is NaN (i.e., was -1).
# Different pathologies can have different valid-row sets.
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 7: Label distribution per target pathology (excluding -1)")
print("=" * 70)

def summarize_label(df_in, col):
    """Count pos/neg after excluding -1 and treating NaN as negative."""
    v = df_in[col].copy()
    # First: -1 means "exclude from eval"
    excluded = (v == -1.0).sum()
    # Remaining: NaN -> 0 (negative), 1.0 -> 1 (positive), 0.0 -> 0
    valid_mask = v != -1.0
    v_valid = v[valid_mask].fillna(0.0)
    pos = int((v_valid == 1.0).sum())
    neg = int((v_valid == 0.0).sum())
    return {"pos": pos, "neg": neg, "excluded_uncertain": int(excluded),
            "total_valid": pos + neg}

for p in TARGET_PATHOLOGIES:
    s = summarize_label(df, p)
    flag = "✓" if s["pos"] >= MIN_POSITIVES_FOR_AUC else "✗ TOO FEW"
    print(f"  {p:20s}  pos={s['pos']:>5}  neg={s['neg']:>5}  "
          f"uncertain_excluded={s['excluded_uncertain']:>3}  [{flag}]")


# -----------------------------------------------------------------------------
# Step 8: Split into shared-only vs extra-pathology subsets
# -----------------------------------------------------------------------------
# Definition:
#   shared-only: all 12 extra pathology labels are either 0.0 or NaN (treated 0)
#   extra-pathology: at least one extra label is 1.0
# (Uncertain -1 in extra labels is ambiguous; we conservatively treat -1 as
#  "not confidently extra" and group them with shared-only. Also report how
#  many samples this ambiguity affects.)
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 8: Shared-only vs Extra-pathology split")
print("=" * 70)

extra_matrix = df[EXTRA_PATHOLOGIES].copy()

# "Confidently extra" = any extra pathology is 1.0
has_extra_confident = (extra_matrix == 1.0).any(axis=1)

# "Ambiguous extra" = no 1.0, but at least one -1.0
has_extra_uncertain = (~has_extra_confident) & (extra_matrix == -1.0).any(axis=1)

print(f"Confidently extra-pathology (>=1 extra label == 1.0): {int(has_extra_confident.sum())}")
print(f"Ambiguous (no 1.0 but has -1.0 in extras):            {int(has_extra_uncertain.sum())}")
print(f"Clean shared-only (all extras in {{0, NaN}}):         "
      f"{int((~has_extra_confident & ~has_extra_uncertain).sum())}")

# Main split: confidently-extra vs everything else
df_shared = df[~has_extra_confident].copy()
df_extra  = df[ has_extra_confident].copy()

print(f"\n--- Shared-only subset (N={len(df_shared)}) ---")
for p in TARGET_PATHOLOGIES:
    s = summarize_label(df_shared, p)
    flag = "✓" if s["pos"] >= MIN_POSITIVES_FOR_AUC else "✗ TOO FEW"
    print(f"  {p:20s}  pos={s['pos']:>5}  neg={s['neg']:>5}  [{flag}]")

print(f"\n--- Extra-pathology subset (N={len(df_extra)}) ---")
for p in TARGET_PATHOLOGIES:
    s = summarize_label(df_extra, p)
    flag = "✓" if s["pos"] >= MIN_POSITIVES_FOR_AUC else "✗ TOO FEW"
    print(f"  {p:20s}  pos={s['pos']:>5}  neg={s['neg']:>5}  [{flag}]")


# -----------------------------------------------------------------------------
# Step 9: Sanity spot-check — construct an image path for one sample
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 9: Sanity check - image path construction")
print("=" * 70)

def build_path(row):
    subj = str(row["subject_id"])
    return IMAGES_DIR / f"p{subj[:2]}" / f"p{subj}" / f"s{row['study_id']}" / f"{row['dicom_id']}.jpg"

sample_row = df.iloc[0]
sample_path = build_path(sample_row)
print(f"Sample dicom_id:   {sample_row['dicom_id']}")
print(f"Constructed path:  {sample_path}")
print(f"File exists:       {sample_path.exists()}")

# Random sample check across 10 dicoms
missing_path = 0
for _, row in df.sample(min(10, len(df)), random_state=0).iterrows():
    if not build_path(row).exists():
        missing_path += 1
print(f"Random 10-sample path check: {10 - missing_path}/10 files found at expected path")

print("\n" + "=" * 70)
print("Feasibility check complete.")
print("=" * 70)