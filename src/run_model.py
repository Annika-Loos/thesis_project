from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import numpy as np
from .dataio import DatasetConfig, load_processed, split_df
from .text import ensure_text_column, ensure_numeric_feature_columns, resolve_cv_json_dir
from .evaluation import compute_performance, fairness_summary
from .paths import RESULTS_DIR, PROCESSED_DIR
from .models.xgb_baseline import XGBTextBaseline, XGBConfig
from .models.bert_baseline import BertBaseline, BertBaselineConfig
from .models.jobbert_baseline import JobBertBaseline, JobBertBaselineConfig
from .mitigation import compute_reweighing_weights, apply_massaging

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgb","bert","jobbert"], required=True)
    ap.add_argument("--label-col", default="label_unbiased")
    ap.add_argument("--metadata-csv", default="cv_metadata_with_labels_two_worlds.csv")
    ap.add_argument("--split-csv", default="train_val_test_split_two_worlds.csv")
    ap.add_argument("--outdir", default=str(RESULTS_DIR))
    ap.add_argument("--cv-dir", default=None)
    ap.add_argument("--mitigation", choices=["none", "reweight", "massaging"], default="none")
    
    args = ap.parse_args()
    variant = args.mitigation

    dcfg = DatasetConfig(metadata_csv=args.metadata_csv, split_csv=args.split_csv, label_col=args.label_col)
    df = load_processed(dcfg, processed_dir=PROCESSED_DIR)

    cv_dir = resolve_cv_json_dir(Path(args.cv_dir) if args.cv_dir else None)
    # Metadata CSV contains only IDs + labels; build model features from CV JSONs.
    df = ensure_text_column(df, filename_col=dcfg.filename_col, text_col=dcfg.text_col, cv_dir=cv_dir)
    df = ensure_numeric_feature_columns(df, filename_col=dcfg.filename_col, cv_dir=cv_dir)


    train = split_df(df, "train"); val = split_df(df, "val"); test = split_df(df, "test")
    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)


    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sample_weights = None
    if args.mitigation == "reweight":
        sample_weights = compute_reweighing_weights(train, label_col=dcfg.label_col)

    if args.mitigation == "massaging":
        train = apply_massaging(
            train,
            label_col=dcfg.label_col,
            privileged_val="Woman",      
            unprivileged_val="Man",      
        )
    if args.model == "xgb":
        model = XGBTextBaseline(XGBConfig(text_col=dcfg.text_col))
        model.fit(train, train[dcfg.label_col].astype(int).to_numpy(), sample_weight=sample_weights)
    elif args.model == "bert":
        model = BertBaseline(BertBaselineConfig())
        model.fit(train, val, label_col=dcfg.label_col, sample_weights=sample_weights)
    else:
        model = JobBertBaseline(JobBertBaselineConfig())
        model.fit(train, val, label_col=dcfg.label_col, sample_weights=sample_weights)



    val_proba = model.predict_proba(val)
    val_pred = (val_proba >= 0.5).astype(int)

    test_proba = model.predict_proba(test)
    test_pred = (test_proba >= 0.5).astype(int)

    #proba = model.predict_proba(test)
    #pred = (proba >= 0.5).astype(int)
    y_true = test[dcfg.label_col].astype(int).to_numpy()
    perf = compute_performance(y_true, test_pred, test_proba)
    print("Performance (test):", perf)

    out = test[[dcfg.filename_col, dcfg.sensitive_attr_col, dcfg.sensitive_value_col, "split", dcfg.label_col]].copy()
    out["y_proba"] = test_proba
    out["y_pred"] = test_pred

    out_val = val[[dcfg.filename_col, dcfg.sensitive_attr_col, dcfg.sensitive_value_col, "split", dcfg.label_col]].copy()
    out_val["y_proba"] = val_proba
    out_val["y_pred"] = val_pred

    val_pred_path = outdir / f"{args.model}_val_predictions_{dcfg.label_col}_{variant}.csv"
    out_val.to_csv(val_pred_path, index=False)

    pred_path = outdir / f"{args.model}_test_predictions_{dcfg.label_col}_{variant}.csv"
    out.to_csv(pred_path, index=False)

    fairness_rows = []
    attr_col = dcfg.sensitive_attr_col
    val_col = dcfg.sensitive_value_col

    for attr, sub in out.groupby(attr_col):
        # how many distinct values exist for this attribute (within its own CVs)?
        n_values = sub[val_col].astype(str).nunique()

        if n_values <= 1:
            # --- signal vs rest on FULL test set ---
            tmp = out.copy()
            tmp["label"] = tmp[dcfg.label_col].astype(int)

            # group = signal value for those CVs that carry this attribute, else REST
            tmp["__group__"] = np.where(
                tmp[attr_col].astype(str).to_numpy() == str(attr),
                tmp[val_col].astype(str).to_numpy(),
                "REST"
            )

            fs = fairness_summary(
                tmp,
                label_col="label",
                pred_col="y_pred",
                group_col="__group__",
                reference="REST",
                sensitive_attribute=attr,
            )
            fs.insert(0, "sensitive_attribute", attr)
            fairness_rows.append(fs)

        else:
            # --- multi-valued: compute fairness within-attribute only (as before) ---
            sub2 = sub.copy()
            sub2["label"] = sub2[dcfg.label_col].astype(int)

            fs = fairness_summary(
                sub2,
                label_col="label",
                pred_col="y_pred",
                group_col=val_col,
                reference=None,
                sensitive_attribute=attr,
            )
            fs.insert(0, "sensitive_attribute", attr)
            fairness_rows.append(fs)

    fair = pd.concat(fairness_rows, ignore_index=True)


    fair_path = outdir / f"{args.model}_fairness_{dcfg.label_col}_{variant}.csv"
    fair.to_csv(fair_path, index=False)

    summ = pd.DataFrame([{ "model": args.model, "label_col": dcfg.label_col, **asdict(perf)}])
    summ_path = outdir / f"{args.model}_metrics_{dcfg.label_col}_{variant}.csv"
    summ.to_csv(summ_path, index=False)

    print("Wrote:", pred_path)
    print("Wrote:", fair_path)
    print("Wrote:", summ_path)
    print("Wrote:", val_pred_path)

if __name__ == "__main__":
    main()
