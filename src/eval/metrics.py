import warnings
from functools import partial

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_ece(y_true, y_prob, n_bins=15):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True)
    bin_indices = np.clip(bin_indices, 1, n_bins)

    ece = 0.0
    n_samples = len(y_true)
    for b in range(1, n_bins + 1):
        mask = bin_indices == b
        count = mask.sum()
        if count == 0:
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (count / n_samples) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_brier_score(y_true, y_prob):
    """Brier Score — mean squared error between predicted probability and true label."""
    return float(np.mean((y_prob - y_true) ** 2))


def compute_confidence_on_incorrect(y_true, y_prob, threshold=0.5):
    """Average model confidence on incorrectly predicted samples."""
    y_pred = (y_prob >= threshold).astype(int)
    incorrect = y_pred != y_true
    if not incorrect.any():
        return 0.0
    confidence = np.maximum(y_prob[incorrect], 1.0 - y_prob[incorrect])
    return float(confidence.mean())


def compute_reliability_diagram_data(y_true, y_prob, n_bins=15):
    """Return binned data for plotting a reliability diagram."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_indices = np.digitize(y_prob, bin_edges, right=True)
    bin_indices = np.clip(bin_indices, 1, n_bins)

    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(1, n_bins + 1):
        mask = bin_indices == b
        count = mask.sum()
        bin_counts[b - 1] = count
        if count > 0:
            bin_accuracies[b - 1] = y_true[mask].mean()
            bin_confidences[b - 1] = y_prob[mask].mean()

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_metric(metric_fn, preds, labels, n_iter=1000,
                     confidence=0.95, seed=42):
    """Compute a bootstrap confidence interval for any scalar metric.

    Args:
        metric_fn: callable(preds, labels) -> float
        preds:     1-D numpy array of predicted probabilities
        labels:    1-D numpy array of ground-truth 0/1 labels
        n_iter:    number of bootstrap resamples
        confidence: CI level (0.95 → 95 %)
        seed:      random seed for reproducibility

    Returns:
        dict with keys 'point', 'lower', 'upper', 'std'
    """
    point = float(metric_fn(preds, labels))

    rng = np.random.RandomState(seed)
    n = len(preds)
    scores = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        try:
            scores.append(metric_fn(preds[idx], labels[idx]))
        except ValueError:
            # e.g. roc_auc_score fails when a resample has only one class
            continue

    scores = np.array(scores)
    scores = scores[np.isfinite(scores)]
    alpha = 1.0 - confidence
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))

    return {"point": point, "lower": lower, "upper": upper,
            "std": float(scores.std())}


try:
    from sklearn.exceptions import UndefinedMetricWarning
except ImportError:
    UndefinedMetricWarning = UserWarning


def _safe_auc(preds, labels):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        return roc_auc_score(labels, preds)


def auc_with_ci(preds, labels, **kwargs):
    """ROC-AUC with bootstrap confidence interval."""
    return bootstrap_metric(_safe_auc, preds, labels, **kwargs)


def ece_with_ci(preds, labels, n_bins=15, **kwargs):
    """ECE with bootstrap confidence interval."""
    fn = partial(compute_ece, n_bins=n_bins)
    return bootstrap_metric(
        lambda p, l: fn(l, p),
        preds, labels, **kwargs,
    )
