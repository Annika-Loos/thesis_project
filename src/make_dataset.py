
"""
End-to-end dataset build script.

Reads
- data/findhr_synthetic_cv_dataset-ver20251203/semisynthetic_cv/semisynthetic_cv_list.csv
- data/findhr_synthetic_cv_dataset-ver20251203/semisynthetic_cv/json_format/*.json

Writes
- data/processed/cv_metadata_with_labels.csv        (includes extracted features + label_unbiased + optional biased labels)
- data/processed/train_val_test_split.csv           (split mapping by filename)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.labeling import BiasConfig, add_biased_labels, build_labeled_metadata, label_diagnostics, group_selection_rates
from src.splitting import SplitConfig, split_train_val_test

def main() -> None:
    data_root = Path("data/findhr_synthetic_cv_dataset-ver20251203")
    meta_path = data_root / "semisynthetic_cv" / "semisynthetic_cv_list.csv"
    cv_dir = data_root / "semisynthetic_cv" / "json_format"

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    df_meta = pd.read_csv(meta_path)

    # Unbiased labeling world (default)
    df_labeled = build_labeled_metadata(
        df_meta,
        cv_dir=cv_dir,
        group_cols=("sector", "years_professional_experience"),
        selectivity=0.30,
        normalize="global_max",   # matches your original notebook
        noise_std=0.02,           # small noise to reduce ties + add realism
        seed=12229942,
    )

    # Optional controlled biased world(s) for mitigation experiments
    attr_col, val_col = "demographic_var", "demographic_var_value"
    print(f"[make_dataset] using sensitive columns: {attr_col=} {val_col=}")

    # Two controlled biased worlds: penalize minority==Yes with two strengths
    bias_cfgs = [
        BiasConfig(
            name="minority_p05",
            match_cols={attr_col: "minority", val_col: "Yes"},
            penalty=0.05,
        ),
        BiasConfig(
            name="minority_p10",
            match_cols={attr_col: "minority", val_col: "Yes"},
            penalty=0.10,
        ),
    ]

    for bias_cfg in bias_cfgs:
        try:
            df_labeled = add_biased_labels(
                df_labeled,
                bias_cfg=bias_cfg,
                group_cols=("sector", "years_professional_experience"),
                selectivity=0.30,
                normalize="global_max",
                noise_std=0.02,
                seed=12229942,
            )
            print(f"[make_dataset] added biased labels: {bias_cfg.name}")
        except Exception as e:
            print(f"[make_dataset] NOTE: Failed to add biased labels ({bias_cfg.name}): {e}")

    out_labeled = processed_dir / "cv_metadata_with_labels_two_worlds.csv"
    df_labeled.to_csv(out_labeled, index=False)
    print(f"[make_dataset] wrote: {out_labeled}")

    # Diagnostics (always for unbiased labels)
    diag = label_diagnostics(df_labeled, label_col="label_unbiased")
    print("[make_dataset] unbiased shortlist rate summary (first 10 rows):")
    print(diag.head(10).to_string(index=False))

    # If we created a biased label, print demographic selection rates
    attr_col, val_col = "demographic_var", "demographic_var_value"

    for lab in ["label_biased_minority_p05", "label_biased_minority_p10"]:
        if lab in df_labeled.columns:
            print(f"[make_dataset] selection rates by demographic group (unbiased) within {attr_col,val_col}:")
            print(
                group_selection_rates(df_labeled, "label_unbiased", [attr_col, val_col])
                .sort_values("shortlist_rate")
                .to_string(index=False)
            )
            print(f"[make_dataset] selection rates by demographic group ({lab}) within {attr_col,val_col}:")
            print(
                group_selection_rates(df_labeled, lab, [attr_col, val_col])
                .sort_values("shortlist_rate")
                .to_string(index=False)
            )
    # Splits (70/10/20) 
    cfg = SplitConfig(train_frac=0.70, val_frac=0.10, test_frac=0.20, seed=12229942)
    # split uses label_unbiased by default; pass label_col=... if you want biased-world splits later
    splits = split_train_val_test(df_labeled, cfg=cfg, label_col="label_unbiased")

    out_splits = processed_dir / "train_val_test_split_two_worlds.csv"
    splits.to_csv(out_splits, index=False)
    print(f"[make_dataset] wrote: {out_splits}")

    merged = df_labeled.merge(splits, on="filename", how="inner")
    print("[make_dataset] split counts:")
    print(merged["split"].value_counts().to_string())
    print("[make_dataset] label_unbiased proportions by split:")
    print(merged.groupby("split")["label_unbiased"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
