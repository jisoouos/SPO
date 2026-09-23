from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_eer(scores, labels):
    """
    Compute Equal Error Rate (EER).

    Args:
        scores (torch.Tensor or array-like): Score values.
        labels (list or array-like): Ground truth labels (1: target, 0: non-target).

    Returns:
        float: EER value (in percentage).
    """
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    # Interpolate between FPR and TPR to compute EER
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100
