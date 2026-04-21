from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from .evaluation import compute_performance, fairness_summary


GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def selection_rate(y_pred: np.ndarray) -> float:
    if len(y_pred) == 0:
        return np.nan
    return float(np.mean(y_pred == 1))


def tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = y_true == 1
    if pos.sum() == 0:
        return np.nan
    return float(np.mean(y_pred[pos] == 1))


def apply_group_thresholds(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    target_attr: str,
    proba_col: str = "y_proba",
    attr_col: str = "demographic_var",
    val_col: str = "demographic_var_value",
    default_threshold: float = 0.5,
) -> np.ndarray:
    """
    Apply group-specific thresholds only for rows where demographic_var == target_attr.
    All other rows use default_threshold.
    """
    proba = df[proba_col].astype(float).to_numpy()
    y_pred = (proba >= default_threshold).astype(int)

    is_target_attr = df[attr_col].astype(str) == target_attr
    sub = df.loc[is_target_attr].copy()

    for group_value, thr in thresholds.items():
        mask = is_target_attr & (df[val_col].astype(str) == group_value)
        y_pred[mask.to_numpy()] = (
            df.loc[mask, proba_col].astype(float).to_numpy() >= thr
        ).astype(int)

    return y_pred


def evaluate_dp_gap(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    target_attr: str,
    attr_col: str = "demographic_var",
    val_col: str = "demographic_var_value",
) -> float:
    sub = df[df[attr_col].astype(str) == target_attr].copy()
    if sub.empty:
        return np.inf

    yp = pd.Series(y_pred, index=df.index).loc[sub.index].to_numpy()
    groups = sorted(sub[val_col].astype(str).dropna().unique())

    if len(groups) < 2:
        return np.inf

    rates = []
    for g in groups:
        mask = sub[val_col].astype(str) == g
        sr = selection_rate(yp[mask.to_numpy()])
        if np.isnan(sr):
            return np.inf
        rates.append(sr)

    return float(max(rates) - min(rates))

def evaluate_eo_gap(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    label_col: str,
    target_attr: str,
    attr_col: str = "demographic_var",
    val_col: str = "demographic_var_value",
) -> float:
    sub = df[df[attr_col].astype(str) == target_attr].copy()
    if sub.empty:
        return np.inf

    yp = pd.Series(y_pred, index=df.index).loc[sub.index].to_numpy()
    yt = sub[label_col].astype(int).to_numpy()
    groups = sorted(sub[val_col].astype(str).dropna().unique())

    if len(groups) < 2:
        return np.inf

    tprs = []
    for g in groups:
        mask = sub[val_col].astype(str) == g
        tpr_g = tpr(yt[mask.to_numpy()], yp[mask.to_numpy()])
        if np.isnan(tpr_g):
            return np.inf
        tprs.append(tpr_g)

    return float(max(tprs) - min(tprs))


def performance_penalty(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    label_col: str,
) -> float:
    """
    Small penalty term so that among equally fair threshold pairs,
    we prefer the one with better overall F1.
    """
    yt = df[label_col].astype(int).to_numpy()
    perf = compute_performance(yt, y_pred, y_proba=None)
    if np.isnan(perf.f1):
        return 1.0
    return 1.0 - perf.f1


from itertools import product

def find_best_thresholds(
    df_val: pd.DataFrame,
    label_col: str,
    objective: str,
    target_attr: str,
    proba_col: str = "y_proba",
    attr_col: str = "demographic_var",
    val_col: str = "demographic_var_value",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Grid search over thresholds for all values of one selected attribute.
    objective:
      - 'dp': minimize selection-rate gap
      - 'eo': minimize TPR gap
    """
    sub = df_val[df_val[attr_col].astype(str) == target_attr].copy()
    groups = sorted(sub[val_col].astype(str).dropna().unique())

    if len(groups) < 2:
        raise ValueError(f"Attribute '{target_attr}' has fewer than 2 groups in validation data.")

    rows: List[Dict] = []
    best_score = np.inf
    best_thresholds = {g: 0.5 for g in groups}

    for combo in product(GRID, repeat=len(groups)):
        thresholds = dict(zip(groups, combo))

        y_pred = apply_group_thresholds(
            df_val,
            thresholds=thresholds,
            target_attr=target_attr,
            proba_col=proba_col,
            attr_col=attr_col,
            val_col=val_col,
        )

        if objective == "dp":
            fairness_gap = evaluate_dp_gap(
                df_val,
                y_pred,
                target_attr=target_attr,
                attr_col=attr_col,
                val_col=val_col,
            )
        elif objective == "eo":
            fairness_gap = evaluate_eo_gap(
                df_val,
                y_pred,
                label_col=label_col,
                target_attr=target_attr,
                attr_col=attr_col,
                val_col=val_col,
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")

        penalty = performance_penalty(df_val, y_pred, label_col=label_col)

        score = fairness_gap + 0.01 * penalty

        row = {
            "fairness_gap": fairness_gap,
            "performance_penalty": penalty,
            "score": score,
        }
        for g, thr in thresholds.items():
            row[f"thr_{g}"] = thr
        rows.append(row)

        if score < best_score:
            best_score = score
            best_thresholds = thresholds.copy()

    trace = pd.DataFrame(rows).sort_values(["score", "fairness_gap", "performance_penalty"])
    return best_thresholds, trace


def compute_and_save_reports(
    df_test_out: pd.DataFrame,
    outdir: Path,
    model: str,
    label_col: str,
    suffix: str,
) -> None:
    y_true = df_test_out[label_col].astype(int).to_numpy()
    y_pred = df_test_out["y_pred"].astype(int).to_numpy()

    y_proba = df_test_out["y_proba"].astype(float).to_numpy()
    perf = compute_performance(y_true, y_pred, y_proba=y_proba)
    perf_df = pd.DataFrame(
        [
            {
                "model": model,
                "label_col": label_col,
                "variant": suffix,
                "accuracy": perf.accuracy,
                "precision": perf.precision,
                "recall": perf.recall,
                "f1": perf.f1,
                "roc_auc": perf.roc_auc,
                "pr_auc": perf.pr_auc,
            }
        ]
    )
    perf_path = outdir / f"{model}_metrics_{label_col}_{suffix}.csv"
    perf_df.to_csv(perf_path, index=False)

    fairness_rows = []
    for attr in sorted(df_test_out["demographic_var"].astype(str).unique()):
        sub = df_test_out[df_test_out["demographic_var"].astype(str) == attr].copy()

        # signal-vs-rest for religion
        if attr == "religion":
            for signal in sorted(sub["demographic_var_value"].astype(str).dropna().unique()):
                tmp = sub.copy()
                tmp["__group__"] = np.where(
                    tmp["demographic_var_value"].astype(str) == signal,
                    signal,
                    "REST",
                )
                fs = fairness_summary(
                    tmp,
                    label_col=label_col,
                    pred_col="y_pred",
                    group_col="__group__",
                    reference="REST",
                    sensitive_attribute=attr,
                )
                fs["sensitive_attribute"] = attr
                fs["analysis_type"] = "signal_vs_rest"
                fs["signal_group"] = signal
                fairness_rows.append(fs)
        else:
            fs = fairness_summary(
                sub,
                label_col=label_col,
                pred_col="y_pred",
                group_col="demographic_var_value",
                reference=None,
                sensitive_attribute=attr,
            )
            fs["sensitive_attribute"] = attr
            fs["analysis_type"] = "groupwise"
            fs["signal_group"] = None
            fairness_rows.append(fs)

    fairness_df = pd.concat(fairness_rows, ignore_index=True)
    fairness_df["model"] = model
    fairness_df["label_col"] = label_col
    fairness_df["variant"] = suffix

    fairness_path = outdir / f"{model}_fairness_{label_col}_{suffix}.csv"
    fairness_df.to_csv(fairness_path, index=False)

    print("Wrote:", perf_path)
    print("Wrote:", fairness_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--model", required=True, choices=["xgb", "bert", "jobbert"])
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--objective", required=True, choices=["dp", "eo"])
    ap.add_argument("--attribute", required=True, help="Sensitive attribute to optimize thresholds for")

    ap.add_argument("--val-file", default=None)
    ap.add_argument("--test-file", default=None)

    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    val_file = (
        Path(args.val_file)
        if args.val_file is not None
        else results_dir / f"{args.model}_val_predictions_{args.label_col}_none.csv"
    )
    test_file = (
        Path(args.test_file)
        if args.test_file is not None
        else results_dir / f"{args.model}_test_predictions_{args.label_col}_none.csv"
    )

    if not val_file.exists():
        raise FileNotFoundError(f"Validation predictions not found: {val_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test predictions not found: {test_file}")

    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    required_cols = {
        args.label_col,
        "y_proba",
        "demographic_var",
        "demographic_var_value",
    }
    missing_val = required_cols - set(df_val.columns)
    missing_test = required_cols - set(df_test.columns)
    if missing_val:
        raise ValueError(f"Validation file missing columns: {missing_val}")
    if missing_test:
        raise ValueError(f"Test file missing columns: {missing_test}")

    thresholds, trace = find_best_thresholds(
    df_val=df_val,
    label_col=args.label_col,
    objective=args.objective,
    target_attr=args.attribute,
    proba_col="y_proba",
    attr_col="demographic_var",
    val_col="demographic_var_value",
)

    print(f"Best thresholds for attribute={args.attribute}, objective={args.objective}:")
    for g, thr in thresholds.items():
        print(f"  {g} -> {thr}")

    trace_path = results_dir / f"{args.model}_threshold_search_{args.label_col}_{args.attribute}_{args.objective}.csv"
    trace.to_csv(trace_path, index=False)
    print("Wrote:", trace_path)

    df_test_out = df_test.copy()
    df_test_out["y_pred"] = apply_group_thresholds(
        df_test_out,
        thresholds=thresholds,
        target_attr=args.attribute,
        proba_col="y_proba",
        attr_col="demographic_var",
        val_col="demographic_var_value",
    )
    df_test_out["postprocess_objective"] = args.objective
    df_test_out["postprocess_attribute"] = args.attribute

    for g, thr in thresholds.items():
        safe_g = str(g).replace(" ", "_").replace("/", "_")
        df_test_out[f"threshold_{args.attribute}_{safe_g}"] = thr

    suffix = f"threshold_{args.attribute}_{args.objective}"
    out_pred_path = results_dir / f"{args.model}_test_predictions_{args.label_col}_{suffix}.csv"
    df_test_out.to_csv(out_pred_path, index=False)
    print("Wrote:", out_pred_path)

    compute_and_save_reports(
        df_test_out=df_test_out,
        outdir=results_dir,
        model=args.model,
        label_col=args.label_col,
        suffix=suffix,
    )


if __name__ == "__main__":
    main()