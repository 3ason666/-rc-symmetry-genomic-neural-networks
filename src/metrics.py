from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def classification_metrics(labels, probabilities, loss: float, threshold: float = 0.5) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    result = {"loss": float(loss), "accuracy": float(accuracy_score(y, pred)),
              "f1": float(f1_score(y, pred, zero_division=0)), "threshold": float(threshold)}
    for name, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
        try:
            result[name] = float(fn(y, p))
        except ValueError as exc:
            warnings.warn(f"{name} is undefined: {exc}")
            result[name] = np.nan
    return result


def prediction_consistency_metrics(p_forward, p_rc, threshold: float = 0.5) -> tuple[dict, dict]:
    pf, pr = np.asarray(p_forward, float), np.asarray(p_rc, float)
    differences = np.abs(pf - pr)
    flips = (pf >= threshold) != (pr >= threshold)
    per_sample = {"prediction_difference": differences,
                  "prediction_consistency": 1.0 - differences, "prediction_flip": flips.astype(int)}
    summary = {
        "prediction_mean_absolute_difference": float(np.mean(differences)),
        "prediction_median_absolute_difference": float(np.median(differences)),
        "prediction_std_absolute_difference": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0,
        "prediction_p95_absolute_difference": float(np.quantile(differences, 0.95)),
        "prediction_pearson": safe_similarity(pf, pr, "pearson")[0],
        "prediction_spearman": safe_similarity(pf, pr, "spearman")[0],
        "symmetry_flip_rate": float(np.mean(flips)),
    }
    return summary, per_sample


def safe_similarity(a, b, method: str) -> tuple[float, str | None]:
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if x.shape != y.shape or x.size == 0:
        return np.nan, "shape_or_empty"
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan, "non_finite"
    if method in {"pearson", "spearman"} and (np.ptp(x) == 0 or np.ptp(y) == 0):
        return np.nan, "constant_vector"
    if method == "cosine":
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return (np.nan, "zero_vector") if denom == 0 else (float(np.dot(x, y) / denom), None)
    value = pearsonr(x, y).statistic if method == "pearson" else spearmanr(x, y).statistic
    return float(value), None


def top_k_overlap(a, b, k: int) -> float:
    x, y = np.asarray(a), np.asarray(b)
    k = min(int(k), x.size)
    ix = set(np.argpartition(np.abs(x), -k)[-k:])
    iy = set(np.argpartition(np.abs(y), -k)[-k:])
    return len(ix & iy) / k


def attribution_consistency(a, b, k: int) -> tuple[dict, list[str]]:
    result, issues = {"top8_overlap": top_k_overlap(a, b, k)}, []
    for method in ("pearson", "spearman", "cosine"):
        value, issue = safe_similarity(a, b, method)
        result[method] = value
        if issue:
            issues.append(f"{method}:{issue}")
    return result, issues


def motif_localization(scores, motif_start: int, motif_end: int, k: int = 8) -> dict:
    return interval_localization(scores, [(motif_start, motif_end)], k)


def interval_localization(scores, intervals, k: int = 8) -> dict:
    """Localization metrics for the union of one or more half-open intervals."""
    values = np.asarray(scores, float)
    mask = np.zeros(values.size, dtype=bool)
    for start, end in intervals:
        mask[int(start):int(end)] = True
    if not mask.any():
        return {"top8_overlap": np.nan, "mass_fraction": np.nan,
                "vs_background": np.nan, "position_auprc": np.nan}
    k = min(int(k), values.size)
    top = np.argpartition(values, -k)[-k:]
    mass = np.abs(values)
    denom = mass.sum()
    try:
        auprc = float(average_precision_score(mask.astype(int), values))
    except ValueError as exc:
        warnings.warn(f"Position AUPRC undefined: {exc}")
        auprc = np.nan
    return {
        "top8_overlap": float(mask[top].mean()),
        "mass_fraction": float(mass[mask].sum() / denom) if denom > 0 else np.nan,
        "vs_background": float(values[mask].mean() - values[~mask].mean()),
        "position_auprc": auprc,
    }


def describe_columns(frame, group_columns: list[str], value_columns: list[str]):
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, keys))
        for col in value_columns:
            values = group[col].dropna().astype(float)
            rows.append({**base, "metric": col, "n": int(values.size),
                         "mean": values.mean() if len(values) else np.nan,
                         "std": values.std(ddof=1) if len(values) > 1 else np.nan,
                         "median": values.median() if len(values) else np.nan,
                         "q25": values.quantile(.25) if len(values) else np.nan,
                         "q75": values.quantile(.75) if len(values) else np.nan,
                         "p95": values.quantile(.95) if len(values) else np.nan})
    import pandas as pd
    return pd.DataFrame(rows)
