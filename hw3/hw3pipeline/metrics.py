"""Binary log loss with fixed clipping and tie-aware ROC-AUC, as defined in HW3.md."""

from __future__ import annotations

import math
from typing import Any

EPSILON = 1e-12


def log_loss(labels: list[int], probabilities: list[float]) -> float | None:
    if not labels:
        return None
    total = 0.0
    for label, probability in zip(labels, probabilities, strict=True):
        clipped = min(max(probability, EPSILON), 1 - EPSILON)
        total -= label * math.log(clipped) + (1 - label) * math.log(1 - clipped)
    return total / len(labels)


def roc_auc(labels: list[int], probabilities: list[float]) -> float | None:
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    ordered = sorted(zip(probabilities, labels, strict=True))
    rank_sum, index = 0.0, 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[index][0]:
            stop += 1
        average_rank = (index + 1 + stop) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:stop])
        index = stop
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def record(
    experiment_id: str, role: str, split: str, group: str, labels: list[int], probabilities: list[float]
) -> dict[str, Any]:
    n, n_positive = len(labels), sum(labels)
    if n == 0:
        loss_reason, auc_reason = "empty_group", "empty_group"
    else:
        loss_reason = None
        auc_reason = None if 0 < n_positive < n else "single_class"
    return {
        "experiment_id": experiment_id,
        "role": role,
        "split": split,
        "group": group,
        "n": n,
        "n_positive": n_positive,
        "log_loss": log_loss(labels, probabilities),
        "roc_auc": roc_auc(labels, probabilities),
        "log_loss_reason": loss_reason,
        "roc_auc_reason": auc_reason,
    }
