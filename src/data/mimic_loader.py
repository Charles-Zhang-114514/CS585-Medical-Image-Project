"""
MIMIC-CXR Dataset for cross-domain evaluation.

Used as the cross-domain target in CS585 project "When Should We Trust Medical
Image Models?". Models are trained on CheXpert and evaluated on this dataset
without any fine-tuning.

Key design decisions:
  1. Evaluation-only (no training augmentation).
  2. Transform exactly matches CheXpert val transform for consistency.
  3. Returns (image, labels, subset_tag) per sample.
       - labels: shape (2,), order [Pneumothorax, Pleural Effusion],
                 values in {0.0, 1.0, -1.0}. -1.0 means "uncertain" and must
                 be excluded per-pathology at metric-computation time.
       - subset_tag: "shared_only" or "extra_pathology" for Setting 4 analysis.
  4. Filter chain (locked in by scripts/check_mimic_feasibility.py):
       split == "test"
       -> dicom exists on local filesystem
       -> ViewPosition in {PA, AP}
       -> study_id has a matching chexpert label row
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Label taxonomy (must match CheXpertDataset.TARGET_COLS order)
# -----------------------------------------------------------------------------
TARGET_COLS = ["Pneumothorax", "Pleural Effusion"]

# Everything in MIMIC chexpert.csv that isn't a target or a key column.
# A sample is "extra_pathology" if any of these is confidently 1.0.
EXTRA_PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Other", "Pneumonia", "Support Devices",
]


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
def _build_image_path(images_root: Path, subject_id, study_id, dicom_id) -> Path:
    """
    Construct MIMIC-CXR-JPG path:
        {images_root}/p{subj[:2]}/p{subj}/s{study}/{dicom}.jpg
    """
    subj = str(subject_id)
    return (
        images_root
        / f"p{subj[:2]}"
        / f"p{subj}"
        / f"s{study_id}"
        / f"{dicom_id}.jpg"
    )


# -----------------------------------------------------------------------------
# Metadata construction (pure-pandas, no images)
# -----------------------------------------------------------------------------
def build_mimic_test_manifest(
    mimic_root: Path,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build the filtered test-set manifest as a DataFrame.

    This is the deterministic filter chain validated in
    scripts/check_mimic_feasibility.py. Separated from the Dataset class so
    the manifest can be inspected, cached, or reused (e.g. for bootstrap).

    Returns a DataFrame with columns:
        dicom_id, study_id, subject_id, ViewPosition, image_path,
        Pneumothorax, Pleural Effusion, <extra pathologies...>, subset_tag

    subset_tag is one of {"shared_only", "extra_pathology"}.
    """
    mimic_root = Path(mimic_root)
    images_root = mimic_root / "images"

    splits   = pd.read_csv(mimic_root / "mimic-cxr-2.0.0-split.csv")
    labels   = pd.read_csv(mimic_root / "mimic-cxr-2.0.0-chexpert.csv")
    metadata = pd.read_csv(mimic_root / "mimic-cxr-2.0.0-metadata.csv")

    # Filter 1: test split
    df = splits[splits["split"] == "test"].copy()
    if verbose:
        print(f"[manifest] test dicoms in split.csv: {len(df)}")

    # Filter 2: exists locally
    local_dicom_ids = {p.stem for p in images_root.rglob("*.jpg")}
    df = df[df["dicom_id"].isin(local_dicom_ids)].copy()
    if verbose:
        print(f"[manifest] after local-file filter:    {len(df)}")

    # Filter 3: attach ViewPosition, keep frontal
    df = df.merge(
        metadata[["dicom_id", "ViewPosition"]],
        on="dicom_id", how="left",
    )
    df = df[df["ViewPosition"].isin(["PA", "AP"])].copy()
    if verbose:
        print(f"[manifest] after frontal filter:       {len(df)}")

    # Filter 4: attach labels, drop rows without a matching study
    df = df.merge(labels, on=["subject_id", "study_id"], how="left")
    before = len(df)
    df = df.dropna(subset=TARGET_COLS, how="all").copy()
    if verbose:
        print(f"[manifest] after label-join filter:    {len(df)}  "
              f"(dropped {before - len(df)} with no matching labels)")

    # Label cleaning for TARGETS:
    #   NaN -> 0 (negative, blank-means-absent convention)
    #   0.0 stays 0.0, 1.0 stays 1.0, -1.0 stays -1.0 (uncertain, excluded later)
    for col in TARGET_COLS:
        df[col] = df[col].fillna(0.0).astype(np.float32)

    # Subset tag: "extra_pathology" if any EXTRA label is confidently 1.0.
    # Uncertain (-1.0) in extras is NOT counted as extra (conservative).
    extra_mat = df[EXTRA_PATHOLOGIES].fillna(0.0)
    has_extra_confident = (extra_mat == 1.0).any(axis=1)
    df["subset_tag"] = np.where(has_extra_confident, "extra_pathology", "shared_only")

    # Image path (stored as string for DataFrame compatibility)
    df["image_path"] = df.apply(
        lambda r: str(_build_image_path(
            images_root, r["subject_id"], r["study_id"], r["dicom_id"]
        )),
        axis=1,
    )

    df = df.reset_index(drop=True)
    if verbose:
        n_shared = int((df["subset_tag"] == "shared_only").sum())
        n_extra  = int((df["subset_tag"] == "extra_pathology").sum())
        print(f"[manifest] final: {len(df)}  "
              f"(shared_only={n_shared}, extra_pathology={n_extra})")
    return df


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class MIMICCXRDataset(Dataset):
    """
    Cross-domain evaluation Dataset for MIMIC-CXR.

    Each __getitem__ returns (image, labels, subset_tag):
        image:      Tensor, shape (3, 224, 224), ImageNet-normalized
        labels:     Tensor, shape (2,), values in {0.0, 1.0, -1.0}
                    Order: [Pneumothorax, Pleural Effusion]
                    -1.0 means "uncertain" and must be excluded at metric time
        subset_tag: str, "shared_only" or "extra_pathology"

    Args:
        mimic_root:  Directory containing the three MIMIC CSVs and images/
        transform:   Image transform (must match CheXpert val transform for
                     cross-domain comparability)
        subset:      One of {"all", "shared_only", "extra_pathology"}.
                     Filters the dataset to that subset.
        manifest:    Optional pre-built DataFrame from build_mimic_test_manifest.
                     Avoids re-running the filesystem scan if provided.
    """

    TARGET_COLS = TARGET_COLS  # mirror CheXpertDataset.TARGET_COLS

    def __init__(
        self,
        mimic_root,
        transform=None,
        subset: str = "all",
        manifest: pd.DataFrame = None,
    ):
        assert subset in {"all", "shared_only", "extra_pathology"}, \
            f"Unknown subset: {subset}"

        self.mimic_root = Path(mimic_root)
        self.transform = transform
        self.subset = subset

        if manifest is None:
            manifest = build_mimic_test_manifest(self.mimic_root)

        if subset != "all":
            manifest = manifest[manifest["subset_tag"] == subset].copy()
        self.df = manifest.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            [row[col] for col in self.TARGET_COLS],
            dtype=torch.float32,
        )

        return image, labels, row["subset_tag"]

    def __repr__(self):
        n_shared = int((self.df["subset_tag"] == "shared_only").sum())
        n_extra = int((self.df["subset_tag"] == "extra_pathology").sum())
        return (
            f"MIMICCXRDataset(N={len(self.df)}, subset='{self.subset}', "
            f"shared_only={n_shared}, extra_pathology={n_extra})"
        )

    # ---- convenience: label arrays for bootstrap / analysis ----
    def get_label_array(self) -> np.ndarray:
        """
        Returns labels as a (N, 2) numpy array in TARGET_COLS order.
        Values in {0.0, 1.0, -1.0}. Useful for metric computation without
        iterating the dataset.
        """
        return self.df[self.TARGET_COLS].to_numpy(dtype=np.float32)

    def get_subset_array(self) -> np.ndarray:
        """Returns subset_tag as a numpy array of strings, shape (N,)."""
        return self.df["subset_tag"].to_numpy()