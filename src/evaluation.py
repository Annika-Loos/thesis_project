from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

REFERENCE_GROUPS = {
    "minority": "No",
    "perceived_foreign": "No",
    "LGBTQ_status": "No",
    "disability": "No",
    "gender": "Man",
    "age": "<= 30",   # falls du das so beibehalten willst; alternativ größte Gruppe
    "religion": "REST",  # nur relevant bei signal-vs-rest
}

def resolve_reference_group(
    sensitive_attribute: str | None,
    ser: pd.Series,
    explicit_reference: str | None = None,
) -> str:
    if explicit_reference is not None:
        return explicit_reference
    if sensitive_attribute is not None and sensitive_attribute in REFERENCE_GROUPS:
        ref = REFERENCE_GROUPS[sensitive_attribute]
        if ref in set(ser.astype(str)):
            return ref
    return ser.astype(str).value_counts().idxmax()

@dataclass(frozen=True)
class PerformanceMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]

def compute_performance(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> PerformanceMetrics:
    roc_auc = pr_auc = None
    if y_proba is not None and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, y_proba))
        pr_auc = float(average_precision_score(y_true, y_proba))
    return PerformanceMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
    )

def _rate_positive(y: np.ndarray) -> float:
    return float(np.mean(y == 1))

def _tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = y_true == 1
    if pos.sum() == 0:
        return np.nan
    return float(np.mean(y_pred[pos] == 1))

def _fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    neg = y_true == 0
    if neg.sum() == 0:
        return np.nan
    return float(np.mean(y_pred[neg] == 1))

def false_positive_rate_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    reference: Optional[str] = None,
    sensitive_attribute: Optional[str] = None,
) -> Dict[str, float]:
    ser = pd.Series(group).astype(str)
    reference = resolve_reference_group(sensitive_attribute, ser, explicit_reference=reference)

    def fpr(mask):
        yt = y_true[mask]
        yp = y_pred[mask]
        neg = yt == 0
        if neg.sum() == 0:
            return np.nan
        return float(np.mean(yp[neg] == 1))

    ref_fpr = fpr((ser == reference).to_numpy())
    out = {"__reference__": reference}
    for g in ser.unique():
        out[g] = fpr((ser == g).to_numpy()) - ref_fpr if ref_fpr == ref_fpr else np.nan
    return out

def demographic_parity_difference(
    y_pred: np.ndarray,
    group: np.ndarray,
    reference: Optional[str] = None,
    sensitive_attribute: Optional[str] = None,
) -> Dict[str, float]:
    ser = pd.Series(group).astype(str)
    reference = resolve_reference_group(sensitive_attribute, ser, explicit_reference=reference)
    ref_rate = _rate_positive(y_pred[(ser == reference).to_numpy()])
    out = {"__reference__": reference}
    for g in ser.unique():
        out[g] = _rate_positive(y_pred[(ser == g).to_numpy()]) - ref_rate
    return out

def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray,
    group: np.ndarray,
    reference: Optional[str] = None,
    sensitive_attribute: Optional[str] = None,
) -> Dict[str, float]:
    ser = pd.Series(group).astype(str)
    reference = resolve_reference_group(sensitive_attribute, ser, explicit_reference=reference)
    def tpr(mask):
        yt = y_true[mask]; yp = y_pred[mask]
        pos = yt == 1
        if pos.sum() == 0:
            return np.nan
        return float(np.mean(yp[pos] == 1))
    ref_tpr = tpr((ser == reference).to_numpy())
    out = {"__reference__": reference}
    for g in ser.unique():
        out[g] = tpr((ser == g).to_numpy()) - ref_tpr if ref_tpr == ref_tpr else np.nan
    return out

def fairness_summary(
    df_pred: pd.DataFrame,
    label_col: str,
    pred_col: str,
    group_col: str,
    reference: Optional[str] = None,
    sensitive_attribute: Optional[str] = None,
) -> pd.DataFrame:
    y_true = df_pred[label_col].to_numpy()
    y_pred = df_pred[pred_col].to_numpy()
    group = df_pred[group_col].astype(str).to_numpy()

    # allow forcing reference (useful for signal-vs-rest where ref should be REST)
    dpd = demographic_parity_difference(
    y_pred,
    group,
    reference=reference,
    sensitive_attribute=sensitive_attribute,
    )
    ref = dpd["__reference__"]
    eod = equal_opportunity_difference(
        y_true,
        y_pred,
        group,
        reference=ref,
        sensitive_attribute=sensitive_attribute,
    )
    fprd = false_positive_rate_difference(
        y_true,
        y_pred,
        group,
        reference=ref,
        sensitive_attribute=sensitive_attribute,
    )
    rows = []
    for g in sorted(set(group)):
        m = (df_pred[group_col].astype(str).to_numpy() == g)
        yt_g = y_true[m]
        yp_g = y_pred[m]

        rows.append({
            "group": g,
            "n": int(m.sum()),
            "selection_rate": _rate_positive(yp_g),
            "tpr": _tpr(yt_g, yp_g),
            "fpr": _fpr(yt_g, yp_g),
            "dpd_vs_ref": dpd.get(g, np.nan),
            "eod_vs_ref": eod.get(g, np.nan),
            "fprd_vs_ref": fprd.get(g, np.nan),
            "avg_odds_diff_vs_ref": np.nanmean([eod.get(g, np.nan), fprd.get(g, np.nan)]),
            "reference_group": ref,
        })
    return pd.DataFrame(rows)
