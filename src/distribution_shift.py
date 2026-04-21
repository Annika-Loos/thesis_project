from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .evaluation import fairness_summary, compute_performance


def resample_group_to_target_share(
    df: pd.DataFrame,
    attr_name: str,
    target_value: str,
    target_share: float,
    attr_col: str = "demographic_var",
    val_col: str = "demographic_var_value",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Resample rows for one sensitive attribute so that within that
    attribute subset, the target_value has approximately target_share.
    All remaining values of that attribute are treated as the comparison group.
    The rest of the dataset remains unchanged.
    """
    rng = np.random.default_rng(random_state)

    attr_mask = df[attr_col].astype(str) == attr_name
    df_attr = df.loc[attr_mask].copy()
    df_rest = df.loc[~attr_mask].copy()

    if df_attr.empty:
        raise ValueError(f"No rows found for attribute '{attr_name}'")

    df_target = df_attr[df_attr[val_col].astype(str) == target_value].copy()
    df_other = df_attr[df_attr[val_col].astype(str) != target_value].copy()

    if df_target.empty or df_other.empty:
        raise ValueError(
            f"Need both target and non-target groups for attribute '{attr_name}'"
        )

    n_total = len(df_attr)
    n_target_new = int(round(target_share * n_total))
    n_other_new = n_total - n_target_new

    target_idx = rng.choice(df_target.index.to_numpy(), size=n_target_new, replace=True)
    other_idx = rng.choice(df_other.index.to_numpy(), size=n_other_new, replace=True)

    df_attr_shifted = pd.concat(
        [df_target.loc[target_idx], df_other.loc[other_idx]],
        ignore_index=True,
    )

    df_shifted = pd.concat([df_rest, df_attr_shifted], ignore_index=True)
    return df_shifted.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def compute_fairness_reports(
    df_pred: pd.DataFrame,
    label_col: str,
    model: str,
    variant: str,
) -> pd.DataFrame:
    fairness_rows = []

    for attr in sorted(df_pred["demographic_var"].astype(str).unique()):
        sub = df_pred[df_pred["demographic_var"].astype(str) == attr].copy()

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
    fairness_df["variant"] = variant
    return fairness_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--model", required=True, choices=["xgb", "bert", "jobbert"])
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--variant", default="none")
    ap.add_argument("--shift-attr", required=True)
    ap.add_argument("--target-value", required=True)
    ap.add_argument("--target-share", type=float, required=True)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    pred_file = results_dir / f"{args.model}_test_predictions_{args.label_col}_{args.variant}.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    df = pd.read_csv(pred_file)

    shifted = resample_group_to_target_share(
        df=df,
        attr_name=args.shift_attr,
        target_value=args.target_value,
        target_share=args.target_share,
        attr_col="demographic_var",
        val_col="demographic_var_value",
        random_state=args.random_state,
    )

    shift_name = f"shift_{args.shift_attr}_{args.target_value}_{str(args.target_share).replace('.', 'p')}"
    full_variant = f"{args.variant}_{shift_name}"

    pred_out = results_dir / f"{args.model}_test_predictions_{args.label_col}_{full_variant}.csv"
    shifted.to_csv(pred_out, index=False)

    y_true = shifted[args.label_col].astype(int).to_numpy()
    y_pred = shifted["y_pred"].astype(int).to_numpy()
    y_proba = shifted["y_proba"].astype(float).to_numpy()

    perf = compute_performance(y_true, y_pred, y_proba)
    perf_df = pd.DataFrame(
        [{
            "model": args.model,
            "label_col": args.label_col,
            "variant": full_variant,
            "accuracy": perf.accuracy,
            "precision": perf.precision,
            "recall": perf.recall,
            "f1": perf.f1,
            "roc_auc": perf.roc_auc,
            "pr_auc": perf.pr_auc,
        }]
    )
    perf_out = results_dir / f"{args.model}_metrics_{args.label_col}_{full_variant}.csv"
    perf_df.to_csv(perf_out, index=False)

    fairness_df = compute_fairness_reports(
        df_pred=shifted,
        label_col=args.label_col,
        model=args.model,
        variant=full_variant,
    )
    fairness_out = results_dir / f"{args.model}_fairness_{args.label_col}_{full_variant}.csv"
    fairness_df.to_csv(fairness_out, index=False)

    print("Wrote:", pred_out)
    print("Wrote:", perf_out)
    print("Wrote:", fairness_out)


if __name__ == "__main__":
    main()