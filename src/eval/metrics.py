import numpy as np


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
