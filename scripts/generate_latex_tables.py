"""
Generate LaTeX tables for the paper from cross_domain_results.json.

Produces:
    Table 1: Per-pathology AUC across all 5 settings (10 rows = 5 settings x 2 models).
    Table 2: Calibration metrics comparing uncalibrated vs temperature-scaled
             cross-domain results (8 rows = 2 models x 2 pathologies x 2 settings).

Tables are printed to stdout and saved to:
    outputs/tables/auc_table.tex
    outputs/tables/calibration_table.tex

Both tables use the `booktabs` package (\\toprule / \\midrule / \\bottomrule) and
`\\multirow` for readability.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = REPO_ROOT / "outputs" / "analysis" / "cross_domain_results.json"
OUT_DIR = REPO_ROOT / "outputs" / "tables"

SETTINGS = [
    "in_domain",
    "cross_domain_all",
    # cross_domain_ts excluded: AUC is invariant to monotonic transformations
    # (Temperature Scaling scales logits, preserving sample ranking).
    # TS effect is shown in calibration_table.tex instead.
    "cross_domain_shared",
    "cross_domain_extra",
]
SETTING_LABELS = {
    "in_domain": "In-domain (CheXpert)",
    "cross_domain_all": "Cross-domain: all",
    "cross_domain_ts": "Cross-domain: all + TS",
    "cross_domain_shared": "Cross-domain: shared-only",
    "cross_domain_extra": "Cross-domain: extra-pathology",
}

MODELS = ["resnet", "densenet"]
MODEL_LABELS = {"resnet": "ResNet-50", "densenet": "DenseNet-121"}

PATHOLOGIES = ["Pneumothorax", "Pleural Effusion"]


def fmt_ci(metric: dict, digits: int = 3) -> str:
    """Format a metric dict as 'point [lower, upper]' with fixed precision."""
    p = metric["point"]
    lo = metric["lower"]
    hi = metric["upper"]
    return f"{p:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def fmt_n(n_ptx: int, n_pe: int) -> str:
    """Show a single N value when they agree, else 'n_ptx / n_pe'."""
    if n_ptx == n_pe:
        return f"{n_ptx}"
    return f"{n_ptx} / {n_pe}"


def build_auc_table(results: dict) -> str:
    """Table 1 — per-pathology AUC across all 5 settings.

    The tabular is wrapped in \\resizebox{\\textwidth}{!}{...} so the table
    never overflows the page width, and every column is separated by a
    vertical rule for visible borders.
    """
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Per-pathology AUC with 95\% bootstrap confidence intervals "
        r"across all evaluation settings. $N$ is the number of test samples "
        r"(reported as \textit{Pneumothorax\,/\,Pleural Effusion} when the "
        r"two pathologies use different subsets).}"
    )
    lines.append(r"\label{tab:auc_cross_domain}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{|l|l|c|c|c|}")
    lines.append(r"\toprule")
    lines.append(
        r"Setting & Model & Pneumothorax AUC [95\% CI] "
        r"& Pleural Effusion AUC [95\% CI] & $N$ \\"
    )
    lines.append(r"\midrule")

    for s_idx, setting in enumerate(SETTINGS):
        setting_label = SETTING_LABELS[setting]
        for m_idx, model in enumerate(MODELS):
            entry_ptx = results[setting][model]["Pneumothorax"]
            entry_pe = results[setting][model]["Pleural Effusion"]
            auc_ptx = fmt_ci(entry_ptx["auc"])
            auc_pe = fmt_ci(entry_pe["auc"])
            n_str = fmt_n(entry_ptx["n_total"], entry_pe["n_total"])

            if m_idx == 0:
                setting_cell = (
                    r"\multirow{2}{*}{" + setting_label + r"}"
                )
            else:
                setting_cell = ""

            lines.append(
                f"{setting_cell} & {MODEL_LABELS[model]} & {auc_ptx} & {auc_pe} & {n_str} \\\\"
            )
        if s_idx != len(SETTINGS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_calibration_table(results: dict) -> str:
    """Table 2 — calibration metrics: uncalibrated vs TS on cross-domain (all).

    Wrapped in \\resizebox{\\textwidth}{!}{...} so the 6-column layout fits
    within the page width; vertical rules are added for explicit borders.
    """
    calib_settings = [
        ("cross_domain_all", "Uncal."),
        ("cross_domain_ts", "+TS"),
    ]

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Calibration metrics on the full cross-domain MIMIC-CXR test "
        r"set, before and after temperature scaling (TS). Lower is better for "
        r"ECE, Brier, and confidence on incorrect predictions. Values are "
        r"point estimates with 95\% bootstrap confidence intervals.}"
    )
    lines.append(r"\label{tab:calibration_ts}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{|l|l|l|c|c|c|}")
    lines.append(r"\toprule")
    lines.append(
        r"Model & Pathology & Calibration "
        r"& ECE [95\% CI] & Brier [95\% CI] "
        r"& Conf. on Incorrect [95\% CI] \\"
    )
    lines.append(r"\midrule")

    for m_idx, model in enumerate(MODELS):
        for p_idx, path in enumerate(PATHOLOGIES):
            for c_idx, (setting, calib_label) in enumerate(calib_settings):
                entry = results[setting][model][path]
                ece = fmt_ci(entry["ece"])
                brier = fmt_ci(entry["brier"])
                conf = fmt_ci(entry["conf_on_incorrect"])

                if p_idx == 0 and c_idx == 0:
                    model_cell = r"\multirow{4}{*}{" + MODEL_LABELS[model] + r"}"
                else:
                    model_cell = ""

                if c_idx == 0:
                    path_cell = r"\multirow{2}{*}{" + path + r"}"
                else:
                    path_cell = ""

                lines.append(
                    f"{model_cell} & {path_cell} & {calib_label} "
                    f"& {ece} & {brier} & {conf} \\\\"
                )
            if p_idx != len(PATHOLOGIES) - 1:
                lines.append(r"\cmidrule(lr){2-6}")
        if m_idx != len(MODELS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    auc_tex = build_auc_table(results)
    calib_tex = build_calibration_table(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    auc_path = OUT_DIR / "auc_table.tex"
    calib_path = OUT_DIR / "calibration_table.tex"

    auc_path.write_text(auc_tex, encoding="utf-8")
    calib_path.write_text(calib_tex, encoding="utf-8")

    print("% ===== Table 1: AUC across settings =====")
    print(auc_tex)
    print("% ===== Table 2: Calibration (uncal vs TS) =====")
    print(calib_tex)
    print(f"% Saved: {auc_path.relative_to(REPO_ROOT)}")
    print(f"% Saved: {calib_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
