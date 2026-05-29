import numpy as np


def compute_iou_matrix(labels_a: np.ndarray,
                       labels_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between all ROIs in two label images.

    Parameters
    labels_a : np.ndarray
        Integer label image (H, W) from method A.
    labels_b : np.ndarray
        Integer label image (H, W) from method B.

    Returns
    np.ndarray
        IoU matrix of shape (N_a, N_b) where N_a and N_b are the
        number of ROIs in labels_a and labels_b respectively.
    """
    n_a = labels_a.max()
    n_b = labels_b.max()

    if n_a == 0 or n_b == 0:
        return np.zeros((max(n_a, 0), max(n_b, 0)), dtype=np.float32)

    iou = np.zeros((n_a, n_b), dtype=np.float32)

    for i in range(1, n_a + 1):
        mask_a = labels_a == i
        area_a = mask_a.sum()
        if area_a == 0:
            continue
        for j in range(1, n_b + 1):
            mask_b = labels_b == j
            intersection = (mask_a & mask_b).sum()
            if intersection == 0:
                continue
            union = area_a + mask_b.sum() - intersection
            iou[i - 1, j - 1] = intersection / union

    return iou


def match_rois(labels_a: np.ndarray,
               labels_b: np.ndarray,
               iou_threshold: float = 0.3):
    """Match ROIs between two label images using IoU.

    Uses greedy matching: repeatedly picks the highest IoU pair
    above the threshold, then removes both from further matching.

    Parameters
    labels_a : np.ndarray
        Integer label image from method A.
    labels_b : np.ndarray
        Integer label image from method B.
    iou_threshold : float
        Minimum IoU to consider a match.

    Returns
    matched_pairs : list of (int, int)
        List of (id_a, id_b) tuples for matched ROIs.
    only_in_a : list of int
        ROI IDs found only in method A.
    only_in_b : list of int
        ROI IDs found only in method B.
    """
    iou = compute_iou_matrix(labels_a, labels_b)
    n_a, n_b = iou.shape

    matched_pairs = []
    used_a = set()
    used_b = set()

    while True:
        masked = iou.copy()
        for a in used_a:
            masked[a, :] = 0
        for b in used_b:
            masked[:, b] = 0

        best = masked.max()
        if best < iou_threshold:
            break

        idx = np.unravel_index(masked.argmax(), masked.shape)
        a_idx, b_idx = int(idx[0]), int(idx[1])

        matched_pairs.append((a_idx + 1, b_idx + 1))
        used_a.add(a_idx)
        used_b.add(b_idx)

    only_in_a = [i + 1 for i in range(n_a) if i not in used_a]
    only_in_b = [j + 1 for j in range(n_b) if j not in used_b]

    print(f"  ROI matching: {len(matched_pairs)} matched, "
          f"{len(only_in_a)} only in A, {len(only_in_b)} only in B "
          f"(IoU threshold={iou_threshold})")

    return matched_pairs, only_in_a, only_in_b


def compute_f1(labels_pred: np.ndarray,
               labels_gt: np.ndarray,
               iou_threshold: float = 0.5) -> dict:
    """Compute Precision, Recall, F1 against ground truth.

    A predicted ROI is a true positive if it overlaps a ground truth
    ROI with IoU >= threshold. Each ground truth ROI can only match once.

    Parameters
    labels_pred : np.ndarray
        Predicted label image (H, W).
    labels_gt : np.ndarray
        Ground truth label image (H, W).
    iou_threshold : float
        Minimum IoU for a match to count as true positive.

    Returns
    dict
        Dictionary with keys: 'precision', 'recall', 'f1',
        'true_positives', 'false_positives', 'false_negatives'.
    """
    matched, only_pred, only_gt = match_rois(
        labels_pred, labels_gt, iou_threshold=iou_threshold
    )

    tp = len(matched)
    fp = len(only_pred)
    fn = len(only_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }

def global_iou(labels_a, labels_b):
    bin_a = labels_a > 0
    bin_b = labels_b > 0
    inter = (bin_a & bin_b).sum()
    union = (bin_a | bin_b).sum()
    return inter / union if union > 0 else 0.0