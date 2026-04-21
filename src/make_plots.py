from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
MAIN_LABEL = "label_unbiased"
MAIN_VARIANTS = ["none", "reweight", "threshold_dp", "threshold_eo"]
SHIFT_VARIANTS_UNBIASED = [
    "none",
    "none_shift_minority_Yes_0p5",
    "none_shift_minority_Yes_0p7",
]

SHIFT_VARIANTS = [
    "none",
    "none_shift_minority_Yes_0p5",
    "none_shift_minority_Yes_0p7",
]

SHIFT_LABELS = {
    "none": "original",
    "none_shift_minority_Yes_0p5": "shift_50",
    "none_shift_minority_Yes_0p7": "shift_70",
}


def subset_shift_unbiased_fair(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        (df["label_col"] == "label_unbiased") &
        (df["variant"].isin(SHIFT_VARIANTS))
    ].copy()
    return sub


def subset_shift_unbiased_metrics(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        (df["label_col"] == "label_unbiased") &
        (df["variant"].isin(SHIFT_VARIANTS))
    ].copy()
    return sub

sns.set_theme(style="whitegrid")

RESULTS_DIR = Path("results")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_fairness_filename(path: Path) -> dict:
    name = path.stem

    m = re.match(r"^(xgb|bert|jobbert)_fairness_(label_.+)$", name)
    if not m:
        return {}

    model = m.group(1)
    rest = m.group(2)

    known_variants = [
        "none",
        "reweight",
        "threshold_dp",
        "threshold_eo",
    ]

    variant = None
    label_col = None

    for v in known_variants:
        suffix = f"_{v}"
        if rest.endswith(suffix):
            label_col = rest[: -len(suffix)]
            variant = v
            break

    if variant is None:
        for v in known_variants:
            marker = f"_{v}_"
            if marker in rest:
                idx = rest.index(marker)
                label_col = rest[:idx]
                variant = rest[idx + 1 :]
                break

    return {
        "model": model,
        "label_col": label_col,
        "variant": variant,
        "filename": path.name,
    }

def plot_distribution_shift_performance(df_metrics: pd.DataFrame) -> None:
    tmp = subset_shift_unbiased_metrics(df_metrics)

    if tmp.empty:
        print("Skipping plot_distribution_shift_performance: no matching rows found.")
        return

    tmp["scenario"] = tmp["variant"].map(SHIFT_LABELS)
    order = ["original", "shift_50", "shift_70"]

    plt.figure(figsize=(8, 5))

    for model in ["xgb", "bert", "jobbert"]:
        t = tmp[tmp["model"] == model].copy()
        if t.empty:
            continue

        t["scenario"] = pd.Categorical(t["scenario"], categories=order, ordered=True)
        t = t.sort_values("scenario")

        plt.plot(
            t["scenario"],
            t["f1"],
            marker="o",
            linewidth=2,
            label=model,
        )

    plt.ylabel("F1")
    plt.title("Predictive performance under distribution shift (unbiased labels)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_shift_performance_f1_unbiased.png", dpi=300)
    plt.close()


def build_shift_summary_table(df_fair: pd.DataFrame, df_metrics: pd.DataFrame) -> pd.DataFrame:
    fair = subset_shift_unbiased_fair(df_fair)
    fair = fair[
        (fair["sensitive_attribute"] == "minority") &
        (fair["group"].astype(str) == "Yes")
    ].copy()

    metrics = subset_shift_unbiased_metrics(df_metrics).copy()

    if fair.empty or metrics.empty:
        raise ValueError("Shift summary table could not be built: missing fairness or metrics rows.")

    fair_keep = fair[["model", "label_col", "variant", "dpd_vs_ref", "eod_vs_ref"]].copy()
    merged = metrics.merge(
        fair_keep,
        on=["model", "label_col", "variant"],
        how="left",
    )

    merged["scenario"] = merged["variant"].map(SHIFT_LABELS)

    out = merged[[
        "model",
        "scenario",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "dpd_vs_ref",
        "eod_vs_ref",
    ]].copy()

    out["scenario"] = pd.Categorical(
        out["scenario"],
        categories=["original", "shift_50", "shift_70"],
        ordered=True,
    )

    out = out.sort_values(["model", "scenario"]).reset_index(drop=True)

    return out

def export_shift_summary_table(df_fair: pd.DataFrame, df_metrics: pd.DataFrame) -> None:
    table = build_shift_summary_table(df_fair, df_metrics).copy()

    numeric_cols = ["accuracy", "precision", "recall", "f1", "dpd_vs_ref", "eod_vs_ref"]
    for col in numeric_cols:
        table[col] = table[col].round(3)

    table.to_csv(FIG_DIR / "table_shift_summary_unbiased.csv", index=False)
    table.to_latex(
        FIG_DIR / "table_shift_summary_unbiased.tex",
        index=False,
        float_format="%.3f",
    )

    print("Wrote shift summary table.")


def plot_rq2_model_fairness_summary(df_fair: pd.DataFrame, df_metrics: pd.DataFrame) -> None:
    """
    RQ2: grouped bar chart comparing F1, mean|EOD|, mean|DPD| per model.
    Gives a clean model-level summary of the performance-fairness trade-off.
    """
    fair = df_fair[
        (df_fair["label_col"] == MAIN_LABEL) &
        (df_fair["variant"] == "none")
    ].copy()

    if "reference_group" in fair.columns:
        fair = fair[fair["group"].astype(str) != fair["reference_group"].astype(str)]

    metrics = df_metrics[
        (df_metrics["label_col"] == MAIN_LABEL) &
        (df_metrics["variant"] == "none")
    ].copy()

    if fair.empty or metrics.empty:
        print("Skipping plot_rq2_model_fairness_summary: no matching rows.")
        return

    summary = fair.groupby("model").agg(
        mean_abs_eod=("eod_vs_ref", lambda x: x.abs().mean()),
        mean_abs_dpd=("dpd_vs_ref", lambda x: x.abs().mean()),
    ).reset_index()

    summary = summary.merge(
        metrics[["model", "f1"]],
        on="model", how="left"
    )

    model_order = ["xgb", "bert", "jobbert"]
    model_labels = {"xgb": "XGBoost", "bert": "BERT", "jobbert": "JobBERT"}
    summary["model"] = pd.Categorical(summary["model"], categories=model_order, ordered=True)
    summary = summary.sort_values("model")
    summary["model_label"] = summary["model"].map(model_labels)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for ax, col, title, ylim in zip(
        axes,
        ["f1", "mean_abs_eod", "mean_abs_dpd"],
        ["F1-Score", "Mean |EOD|", "Mean |DPD|"],
        [(0.65, 0.86), (0.0, 0.40), (0.0, 0.30)],
    ):
        bars = ax.bar(summary["model_label"], summary[col], color=colors,
                      edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(ylim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, summary[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    axes[0].set_ylabel("Score", fontsize=10)
    axes[1].set_ylabel("Mean Absolute Difference", fontsize=10)
    axes[2].set_ylabel("Mean Absolute Difference", fontsize=10)

    fig.suptitle(
        "Baseline Model Comparison: Predictive Performance vs. Fairness Burden",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_rq2_model_fairness_summary.png", dpi=300)
    plt.close()
    print("Wrote plot_rq2_model_fairness_summary.png")


def plot_rq2_scatter_f1_vs_eod(df_fair: pd.DataFrame, df_metrics: pd.DataFrame) -> None:
    """
    RQ2: scatter plot — one point per (model x attribute), x=F1, y=|EOD|.
    Gender points labelled explicitly. Shows full trade-off landscape.
    """
    fair = df_fair[
        (df_fair["label_col"] == MAIN_LABEL) &
        (df_fair["variant"] == "none")
    ].copy()

    if "reference_group" in fair.columns:
        fair = fair[fair["group"].astype(str) != fair["reference_group"].astype(str)]

    metrics = df_metrics[
        (df_metrics["label_col"] == MAIN_LABEL) &
        (df_metrics["variant"] == "none")
    ].copy()

    if fair.empty or metrics.empty:
        print("Skipping plot_rq2_scatter_f1_vs_eod: no matching rows.")
        return

    # use absolute EOD, one row per model x attribute (mean over groups if >1)
    agg = fair.groupby(["model", "sensitive_attribute"]).agg(
        abs_eod=("eod_vs_ref", lambda x: x.abs().mean()),
        abs_dpd=("dpd_vs_ref", lambda x: x.abs().mean()),
    ).reset_index()

    agg = agg.merge(metrics[["model", "f1"]], on="model", how="left")

    model_styles = {
        "xgb":     {"color": "#2196F3", "marker": "o", "label": "XGBoost"},
        "bert":    {"color": "#FF5722", "marker": "s", "label": "BERT"},
        "jobbert": {"color": "#4CAF50", "marker": "^", "label": "JobBERT"},
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for model, style in model_styles.items():
        sub = agg[agg["model"] == model]
        ax.scatter(
            sub["f1"], sub["abs_eod"],
            s=sub["abs_dpd"] * 1000 + 40,
            color=style["color"],
            marker=style["marker"],
            alpha=0.70,
            edgecolors="white",
            linewidths=0.6,
            label=style["label"],
            zorder=3,
        )
        # label gender points only
        for _, row in sub.iterrows():
            if row["sensitive_attribute"] == "gender":
                ax.annotate(
                    f"gender\n({style['label']})",
                    (row["f1"], row["abs_eod"]),
                    xytext=(7, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=style["color"],
                    fontweight="bold",
                )

    ax.axhline(0.10, color="grey", linestyle="--", linewidth=0.9,
               alpha=0.6, label="|EOD| = 0.10 reference")
    ax.set_xlabel("F1-Score (baseline)", fontsize=12)
    ax.set_ylabel("|Equal Opportunity Difference|", fontsize=12)
    ax.set_title(
        "Fairness–Performance Trade-off: F1 vs |EOD| per Attribute\n"
        "(bubble size proportional to |DPD|)",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_rq2_scatter_f1_eod.png", dpi=300)
    plt.close()
    print("Wrote plot_rq2_scatter_f1_eod.png")

def build_shift_group_error_table(df_fair: pd.DataFrame) -> pd.DataFrame:
    fair = subset_shift_unbiased_fair(df_fair)

    sub = fair[fair["sensitive_attribute"] == "minority"].copy()

    if sub.empty:
        raise ValueError("Group error table could not be built: no minority rows found.")

    keep_cols = [
        c for c in [
            "model",
            "variant",
            "group",
            "selection_rate",
            "tpr",
            "fpr",
            "dpd_vs_ref",
            "eod_vs_ref",
            "fprd_vs_ref",
        ] if c in sub.columns
    ]

    out = sub[keep_cols].copy()
    out["scenario"] = out["variant"].map(SHIFT_LABELS)

    out["scenario"] = pd.Categorical(
        out["scenario"],
        categories=["original", "shift_50", "shift_70"],
        ordered=True,
    )

    order_cols = ["model", "scenario", "group"] + [c for c in keep_cols if c not in ["model", "variant", "group"]]
    out = out.sort_values(["model", "scenario", "group"]).reset_index(drop=True)

    return out[order_cols]

def export_shift_group_error_table(df_fair: pd.DataFrame) -> None:
    table = build_shift_group_error_table(df_fair).copy()

    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].round(3)

    table.to_csv(FIG_DIR / "table_shift_group_errors_minority_unbiased.csv", index=False)
    table.to_latex(
        FIG_DIR / "table_shift_group_errors_minority_unbiased.tex",
        index=False,
        float_format="%.3f",
    )

    print("Wrote shift group-wise error table.")

linestyles = {
    "xgb": "-",
    "bert": "--",
    "jobbert": "--",
}

def compute_performance_impact(df_metrics: pd.DataFrame) -> pd.DataFrame:
    sub = df_metrics[
        df_metrics["label_col"] == "label_unbiased"
    ].copy()

    sub = sub[
        sub["variant"].isin(["none", "reweight", "threshold_dp", "threshold_eo"])
    ]

    base = sub[sub["variant"] == "none"][
        ["model", "f1", "accuracy"]
    ].rename(columns={
        "f1": "f1_base",
        "accuracy": "acc_base"
    })

    other = sub[sub["variant"] != "none"].copy()

    merged = other.merge(base, on="model", how="left")

    merged["delta_f1"] = merged["f1"] - merged["f1_base"]
    merged["delta_acc"] = merged["accuracy"] - merged["acc_base"]

    return merged[["model", "variant", "delta_f1", "delta_acc"]]

def plot_performance_impact(df_metrics: pd.DataFrame):
    df = compute_performance_impact(df_metrics)

    plt.figure(figsize=(8, 5))

    for variant in ["reweight", "threshold_dp", "threshold_eo"]:
        tmp = df[df["variant"] == variant]
        plt.plot(tmp["model"], tmp["delta_f1"], marker="o", label=variant)

    plt.axhline(0, linestyle="--")
    plt.ylabel("Δ F1 (vs baseline)")
    plt.title("Performance impact of mitigation (unbiased labels)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_performance_impact.png", dpi=300)
    plt.close()

def plot_shift_group_error_behavior(df_fair: pd.DataFrame) -> None:
    fair = subset_shift_unbiased_fair(df_fair)
    sub = fair[fair["sensitive_attribute"] == "minority"].copy()

    if sub.empty:
        print("Skipping plot_shift_group_error_behavior: no minority rows found.")
        return

    sub["scenario"] = sub["variant"].map(SHIFT_LABELS)
    scenario_order = ["original", "shift_50", "shift_70"]

    for metric, outname, title in [
        ("tpr", "plot_shift_group_tpr_unbiased.png", "TPR under shift (minority attribute)"),
        ("fpr", "plot_shift_group_fpr_unbiased.png", "FPR under shift (minority attribute)"),
    ]:
        if metric not in sub.columns:
            print(f"Skipping {outname}: column {metric} not found.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

        for i, group in enumerate(["No", "Yes"]):
            ax = axes[i]
            gdata = sub[sub["group"].astype(str) == group].copy()

            for model in ["xgb", "bert", "jobbert"]:
                t = gdata[gdata["model"] == model].copy()
                if t.empty:
                    continue

                t["scenario"] = pd.Categorical(
                    t["scenario"],
                    categories=scenario_order,
                    ordered=True
                )
                t = t.sort_values("scenario")

                ax.plot(
                    t["scenario"],
                    t[metric],
                    marker="o",
                    linewidth=2,
                    linestyle=linestyles[model],
                    label=model,
                )

            ax.set_title(f"group = {group}")
            ax.set_xlabel("scenario")

        axes[0].set_ylabel(metric.upper())
        axes[1].legend()

        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(FIG_DIR / outname, dpi=300)
        plt.close()


def parse_metrics_filename(path: Path) -> dict:
    name = path.stem

    m = re.match(r"^(xgb|bert|jobbert)_metrics_(label_.+)$", name)
    if not m:
        return {}

    model = m.group(1)
    rest = m.group(2)

    known_variants = ["none", "reweight", "threshold_dp", "threshold_eo"]

    variant = None
    label_col = None

    for v in known_variants:
        suffix = f"_{v}"
        if rest.endswith(suffix):
            label_col = rest[: -len(suffix)]
            variant = v
            break

    if variant is None:
        for v in known_variants:
            marker = f"_{v}_"
            if marker in rest:
                idx = rest.index(marker)
                label_col = rest[:idx]
                variant = rest[idx + 1 :]
                break

    if variant is None:
        label_col = rest
        variant = "none"

    return {
        "model": model,
        "label_col": label_col,
        "variant": variant,
        "filename": path.name,
    }


def load_all_fairness() -> pd.DataFrame:
    rows = []

    for path in RESULTS_DIR.glob("*_fairness_*.csv"):
        meta = parse_fairness_filename(path)
        if not meta:
            continue

        df = pd.read_csv(path)

        if "sensitive_attribute" not in df.columns:
            continue
        if "group" not in df.columns:
            continue

        for k, v in meta.items():
            df[k] = v

        rows.append(df)

    if not rows:
        raise ValueError("No fairness CSVs found.")

    out = pd.concat(rows, ignore_index=True)
    out["group"] = out["group"].astype(str)
    out["sensitive_attribute"] = out["sensitive_attribute"].astype(str)

    if "reference_group" in out.columns:
        out["reference_group"] = out["reference_group"].astype(str)

    return out


def load_all_metrics() -> pd.DataFrame:
    rows = []

    for path in RESULTS_DIR.glob("*_metrics_*.csv"):
        meta = parse_metrics_filename(path)
        if not meta:
            continue

        df = pd.read_csv(path)
        for k, v in meta.items():
            df[k] = v
        rows.append(df)

    if not rows:
        raise ValueError("No metrics CSVs found.")

    return pd.concat(rows, ignore_index=True)


def subset_attribute_group(df: pd.DataFrame, attribute: str, group: str) -> pd.DataFrame:
    return df[
        (df["sensitive_attribute"].astype(str) == attribute) &
        (df["group"].astype(str) == str(group))
    ].copy()


def subset_attribute(df: pd.DataFrame, attribute: str) -> pd.DataFrame:
    return df[
        df["sensitive_attribute"].astype(str) == attribute
    ].copy()

PLOT_ATTRIBUTES = ["gender", "age", "minority", "perceived_foreign", "LGBTQ_status","religion", "disability"]


def plot_baseline_fairness(df: pd.DataFrame) -> None:
    sub = df[
        (df["label_col"] == "label_unbiased") &
        (df["variant"] == "none")
    ].copy()

    if "reference_group" in sub.columns:
        sub = sub[sub["group"].astype(str) != sub["reference_group"].astype(str)]

    if sub.empty:
        print("Skipping plot_baseline_fairness: no matching rows found.")
        return

    for attribute in sorted(sub["sensitive_attribute"].astype(str).unique()):
        tmp = sub[sub["sensitive_attribute"].astype(str) == attribute].copy()
        if tmp.empty:
            continue

        pivot = tmp.pivot_table(
            index="model",
            values=["dpd_vs_ref", "eod_vs_ref"],
            aggfunc="mean",
        ).reindex(index=["xgb", "bert", "jobbert"])

        ax = pivot.plot(kind="bar", figsize=(8, 5))
        ax.set_ylabel("Fairness difference")
        ax.set_title(f"Baseline fairness for attribute: {attribute}")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"plot1_baseline_fairness_{attribute}.png", dpi=300)
        plt.close()



def plot_mitigation_comparison_unbiased(df: pd.DataFrame) -> None:
    sub = df[
        (df["label_col"] == "label_unbiased") &
        (df["variant"].isin(["none", "reweight", "threshold_dp", "threshold_eo"]))
    ].copy()

    if "reference_group" in sub.columns:
        sub = sub[sub["group"].astype(str) != sub["reference_group"].astype(str)]

    if sub.empty:
        print("Skipping plot_mitigation_comparison_unbiased: no matching rows found.")
        return

    for attribute in sorted(sub["sensitive_attribute"].astype(str).unique()):
        tmp = sub[sub["sensitive_attribute"].astype(str) == attribute].copy()
        if tmp.empty:
            continue

        grouped = tmp.groupby(["model", "variant"], as_index=False)[["dpd_vs_ref", "eod_vs_ref"]].mean()

        for metric, outname, title in [
            ("dpd_vs_ref", f"plot3_unbiased_mitigation_dpd_{attribute}.png", f"Mitigation comparison on unbiased labels (DPD) - {attribute}"),
            ("eod_vs_ref", f"plot3_unbiased_mitigation_eod_{attribute}.png", f"Mitigation comparison on unbiased labels (EOD) - {attribute}"),
        ]:
            pivot = grouped.pivot(index="model", columns="variant", values=metric)
            pivot = pivot.reindex(
                index=["xgb", "bert", "jobbert"],
                columns=["none", "reweight", "threshold_dp", "threshold_eo"],
            )

            ax = pivot.plot(kind="bar", figsize=(9, 5))
            ax.set_ylabel(metric)
            ax.set_title(title)
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(FIG_DIR / outname, dpi=300)
            plt.close()


def plot_distribution_shift_unbiased(
    df: pd.DataFrame,
    attribute: str,
    group: str,
    keep_variants: list[str],
    variant_labels: dict[str, str],
) -> None:
    sub = df[
        (df["sensitive_attribute"].astype(str) == attribute) &
        (df["group"].astype(str) == str(group))
    ].copy()

    tmp = sub[
        (sub["label_col"] == "label_unbiased") &
        (sub["variant"].isin(keep_variants))
    ].copy()

    if tmp.empty:
        print(f"Skipping plot_distribution_shift_unbiased for {attribute}={group}: no matching rows found.")
        return

    tmp["variant_plot"] = tmp["variant"].map(variant_labels)
    order = list(variant_labels.values())

    for metric, outname_suffix, title_suffix in [
        ("dpd_vs_ref", "dpd", "DPD"),
        ("eod_vs_ref", "eod", "EOD"),
    ]:
        plt.figure(figsize=(8, 5))

        for model in ["xgb", "bert", "jobbert"]:
            t = tmp[tmp["model"] == model].copy()
            if t.empty:
                continue

            t["variant_plot"] = pd.Categorical(
                t["variant_plot"],
                categories=order,
                ordered=True
            )
            t = t.sort_values("variant_plot")

            plt.plot(
                t["variant_plot"],
                t[metric],
                marker="o",
                linewidth=2,
                label=model,
            )

        plt.ylabel(metric)
        plt.title(f"Distribution shift effect on {title_suffix} ({attribute}={group}, unbiased labels)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            FIG_DIR / f"plot4_unbiased_distribution_shift_{title_suffix.lower()}_{attribute}_{group}.png",
            dpi=300
        )
        plt.close()



def plot_bias_heatmaps(df_fair: pd.DataFrame) -> None:
    df = df_fair.copy()

    if "analysis_type" in df.columns:
        df = df[df["analysis_type"].fillna("groupwise") == "groupwise"].copy()

    if "reference_group" in df.columns:
        df = df[df["group"].astype(str) != df["reference_group"].astype(str)].copy()

    sub = df[
        (df["label_col"] == "label_unbiased") &
        (df["variant"] == "none")
    ].copy()

    if sub.empty:
        print("Skipping plot_bias_heatmaps: no matching rows found.")
        return

    for metric, outname, title in [
        ("dpd_vs_ref", "plot_heatmap_baseline_dpd.png", "Baseline bias heatmap (DPD)"),
        ("eod_vs_ref", "plot_heatmap_baseline_eod.png", "Baseline bias heatmap (EOD)"),
    ]:
        if metric not in sub.columns:
            print(f"Skipping {outname}: column {metric} not found.")
            continue

        pivot = sub.pivot_table(
            index="model",
            columns="sensitive_attribute",
            values=metric,
            aggfunc="mean",
        )

        pivot = pivot.reindex(index=["xgb", "bert", "jobbert"])

        plt.figure(figsize=(10, 4.5))
        sns.heatmap(
            pivot,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": metric},
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIG_DIR / outname, dpi=300)
        plt.close()

def plot_fairness_performance_tradeoff_unbiased(df_fair: pd.DataFrame, df_metrics: pd.DataFrame) -> None:
    fair = df_fair[
        (df_fair["label_col"] == "label_unbiased") &
        (df_fair["variant"].isin(["none", "reweight", "threshold_dp", "threshold_eo"]))
    ].copy()

    if "reference_group" in fair.columns:
        fair = fair[fair["group"].astype(str) != fair["reference_group"].astype(str)]

    metrics = df_metrics[
        (df_metrics["label_col"] == "label_unbiased") &
        (df_metrics["variant"].isin(["none", "reweight", "threshold_dp", "threshold_eo"]))
    ].copy()

    if fair.empty or metrics.empty:
        print("Skipping plot_fairness_performance_tradeoff_unbiased: no matching rows found.")
        return

    fair_grouped = fair.groupby(
        ["model", "label_col", "variant", "sensitive_attribute"],
        as_index=False
    )[
        ["dpd_vs_ref", "eod_vs_ref"]
    ].mean()

    merged = fair_grouped.merge(
        metrics[["model", "label_col", "variant", "accuracy", "f1"]],
        on=["model", "label_col", "variant"],
        how="left",
    )

    for attribute in sorted(merged["sensitive_attribute"].astype(str).unique()):
        tmp_attr = merged[merged["sensitive_attribute"].astype(str) == attribute].copy()

        for fairness_metric, perf_metric, outname, title in [
            ("dpd_vs_ref", "f1", f"plot_tradeoff_unbiased_dpd_vs_f1_{attribute}.png", f"Trade-off: DPD vs F1 ({attribute})"),
            ("eod_vs_ref", "f1", f"plot_tradeoff_unbiased_eod_vs_f1_{attribute}.png", f"Trade-off: EOD vs F1 ({attribute})"),
        ]:
            plt.figure(figsize=(8, 6))

            for model in ["xgb", "bert", "jobbert"]:
                t = tmp_attr[tmp_attr["model"] == model].copy()
                if t.empty:
                    continue

                plt.scatter(t[perf_metric], t[fairness_metric], s=90, label=model)

                for _, row in t.iterrows():
                    plt.annotate(
                        row["variant"],
                        (row[perf_metric], row[fairness_metric]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )

            plt.xlabel(perf_metric.upper())
            plt.ylabel(fairness_metric)
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / outname, dpi=300)
            plt.close()

def plot_mitigation_improvement_heatmaps_unbiased(df_fair: pd.DataFrame) -> None:
    fair = df_fair[
        (df_fair["label_col"] == "label_unbiased") &
        (df_fair["variant"].isin(["none", "reweight", "threshold_dp", "threshold_eo"]))
    ].copy()

    if "reference_group" in fair.columns:
        fair = fair[fair["group"].astype(str) != fair["reference_group"].astype(str)]

    if fair.empty:
        print("Skipping plot_mitigation_improvement_heatmaps_unbiased: no matching rows found.")
        return

    grouped = fair.groupby(
        ["model", "variant", "sensitive_attribute"],
        as_index=False
    )[
        ["dpd_vs_ref", "eod_vs_ref"]
    ].mean()

    for attribute in sorted(grouped["sensitive_attribute"].astype(str).unique()):
        tmp = grouped[grouped["sensitive_attribute"].astype(str) == attribute].copy()
        if tmp.empty:
            continue

        base = tmp[tmp["variant"] == "none"][["model", "dpd_vs_ref", "eod_vs_ref"]].copy()
        base = base.rename(columns={
            "dpd_vs_ref": "dpd_base",
            "eod_vs_ref": "eod_base",
        })

        other = tmp[tmp["variant"] != "none"][["model", "variant", "dpd_vs_ref", "eod_vs_ref"]].copy()
        merged = other.merge(base, on="model", how="left")

        merged["dpd_improvement"] = merged["dpd_base"] - merged["dpd_vs_ref"]
        merged["eod_improvement"] = merged["eod_base"] - merged["eod_vs_ref"]

        for value_col, outname, title in [
            ("dpd_improvement", f"plot_heatmap_unbiased_mitigation_dpd_{attribute}.png", f"Mitigation improvement heatmap (DPD) - {attribute}"),
            ("eod_improvement", f"plot_heatmap_unbiased_mitigation_eod_{attribute}.png", f"Mitigation improvement heatmap (EOD) - {attribute}"),
        ]:
            pivot = merged.pivot(index="model", columns="variant", values=value_col)
            pivot = pivot.reindex(
                index=["xgb", "bert", "jobbert"],
                columns=["reweight", "threshold_dp", "threshold_eo"],
            )

            plt.figure(figsize=(7.5, 4.5))
            sns.heatmap(
                pivot,
                annot=True,
                cmap="coolwarm",
                center=0,
                fmt=".2f",
                linewidths=0.5,
                cbar_kws={"label": value_col},
            )
            plt.xticks(rotation=30, ha="right")
            plt.yticks(rotation=0)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(FIG_DIR / outname, dpi=300)
            plt.close()


def plot_bias_across_attributes(df: pd.DataFrame) -> None:
    sub = df[
        (df["label_col"] == "label_unbiased") &
        (df["variant"] == "none")
    ].copy()

    if "reference_group" in sub.columns:
        sub = sub[sub["group"] != sub["reference_group"]]

    if sub.empty:
        print("Skipping plot_bias_across_attributes: no matching rows found.")
        return

    pivot = sub.pivot_table(
        index="model",
        columns="sensitive_attribute",
        values="dpd_vs_ref",
        aggfunc="mean",
    )

    plt.figure(figsize=(10, 5))
    sns.heatmap(
        pivot,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "dpd_vs_ref"},
    )
    plt.title("Bias across sensitive attributes (DPD)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_bias_across_attributes.png", dpi=300)
    plt.close()


def plot_mitigation_performance_comparison(df_metrics: pd.DataFrame) -> None:
    sub = df_metrics[
        (df_metrics["label_col"] == "label_unbiased") &
        (df_metrics["variant"].isin(["none", "reweight", "threshold_dp", "threshold_eo"]))
    ].copy()

    if sub.empty:
        print("Skipping plot_mitigation_performance_comparison: no matching rows found.")
        return

    for metric, outname, title in [
        ("f1", "plot5_unbiased_mitigation_f1.png", "Mitigation comparison on unbiased labels (F1)"),
        ("accuracy", "plot5_unbiased_mitigation_accuracy.png", "Mitigation comparison on unbiased labels (Accuracy)"),
    ]:
        pivot = sub.pivot(index="model", columns="variant", values=metric)
        pivot = pivot.reindex(
            index=["xgb", "bert", "jobbert"],
            columns=["none", "reweight", "threshold_dp", "threshold_eo"],
        )

        ax = pivot.plot(kind="bar", figsize=(9, 5))
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIG_DIR / outname, dpi=300)
        plt.close()

def plot_mitigation_performance_delta(df_metrics: pd.DataFrame) -> None:
    sub = df_metrics[
        (df_metrics["label_col"] == "label_unbiased") &
        (df_metrics["variant"].isin(["none", "reweight", "threshold_dp", "threshold_eo"]))
    ].copy()

    if sub.empty:
        print("Skipping plot_mitigation_performance_delta: no matching rows found.")
        return

    base = sub[sub["variant"] == "none"][["model", "f1", "accuracy"]].rename(
        columns={"f1": "f1_base", "accuracy": "acc_base"}
    )

    other = sub[sub["variant"] != "none"].copy()
    merged = other.merge(base, on="model", how="left")

    merged["delta_f1"] = merged["f1"] - merged["f1_base"]
    merged["delta_accuracy"] = merged["accuracy"] - merged["acc_base"]

    for metric, outname, title in [
        ("delta_f1", "plot5_unbiased_mitigation_delta_f1.png", "Performance impact of mitigation (ΔF1)"),
        ("delta_accuracy", "plot5_unbiased_mitigation_delta_accuracy.png", "Performance impact of mitigation (ΔAccuracy)"),
    ]:
        pivot = merged.pivot(index="model", columns="variant", values=metric)
        pivot = pivot.reindex(
            index=["xgb", "bert", "jobbert"],
            columns=["reweight", "threshold_dp", "threshold_eo"],
        )

        ax = pivot.plot(kind="bar", figsize=(9, 5))
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_ylabel(metric)
        ax.set_title(title)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(FIG_DIR / outname, dpi=300)
        plt.close()

def plot_metric_correlation(df: pd.DataFrame) -> None:
    wanted_cols = ["dpd_vs_ref", "eod_vs_ref", "fprd_vs_ref", "average_odds_difference"]
    cols = [c for c in wanted_cols if c in df.columns]

    if len(cols) < 2:
        print("Skipping plot_metric_correlation: not enough fairness metric columns found.")
        return

    corr = df[cols].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Correlation between fairness metrics")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "plot_metric_correlation.png", dpi=300)
    plt.close()


def main():
    df_fair = load_all_fairness()
    df_metrics = load_all_metrics()

    # Baseline / main unbiased plots
    plot_baseline_fairness(df_fair)
    plot_bias_heatmaps(df_fair)
    plot_bias_across_attributes(df_fair)
    plot_mitigation_performance_comparison(df_metrics)

    # Unbiased mitigation / trade-off / shift
    plot_mitigation_comparison_unbiased(df_fair)
    plot_fairness_performance_tradeoff_unbiased(df_fair, df_metrics)
    plot_distribution_shift_performance(df_metrics)
    plot_mitigation_improvement_heatmaps_unbiased(df_fair)
    plot_performance_impact(df_metrics)
    plot_rq2_model_fairness_summary(df_fair, df_metrics)
    plot_rq2_scatter_f1_vs_eod(df_fair, df_metrics)

    plot_distribution_shift_unbiased(
        df_fair,
        attribute="minority",
        group="Yes",
        keep_variants=[
            "none",
            "none_shift_minority_Yes_0p5",
            "none_shift_minority_Yes_0p7",
        ],
        variant_labels={
            "none": "original",
            "none_shift_minority_Yes_0p5": "shift_50",
            "none_shift_minority_Yes_0p7": "shift_70",
        },
    )

    # Shift tables
    export_shift_summary_table(df_fair, df_metrics)
    export_shift_group_error_table(df_fair)

    # Optional appendix plot
    plot_shift_group_error_behavior(df_fair)

    plot_mitigation_performance_delta(df_metrics)


    print(f"Wrote plots and tables to {FIG_DIR}")


if __name__ == "__main__":
    main()