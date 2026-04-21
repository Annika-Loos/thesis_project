
"""
Labeling utilities for the FINDHR semi-synthetic CV dataset.

What this module does
1) Loads CV JSONs and extracts a small numeric feature set.
2) Computes a qualification score (a transparent proxy screening policy).
3) Assigns labels by selecting the top τ fraction *within* (sector, years_professional_experience) strata.

Why this design
- The dataset has no real hiring decisions; labels must be constructed reproducibly.
- We keep the label policy *independent* of sensitive attributes by default ("unbiased world").
- For bias-mitigation experiments, we additionally support a *controlled biased world* where
  a configurable penalty is applied to specific sensitive groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLS = ["total_exp_months", "n_degrees", "n_skills", "n_jobs"]


@dataclass
class BiasConfig:
    """Configuration for controlled label bias.

    The bias is applied as: score := score - penalty   for rows matching `match_cols`.

    Example
    -------
    BiasConfig(
        name="gender_female",
        match_cols={"sensitive_attribute": "gender", "sensitive_value": "female"},
        penalty=0.05
    )
    """

    name: str
    match_cols: Dict[str, str]
    penalty: float

    def mask(self, df: pd.DataFrame) -> pd.Series:
        m = pd.Series(True, index=df.index)
        for col, val in self.match_cols.items():
            if col not in df.columns:
                raise KeyError(f"BiasConfig refers to missing column: '{col}'")
            m = m & (df[col].astype(str) == str(val))
        return m


def load_cv_features(cv_path: Path) -> Dict[str, float]:
    """Extract a small set of numeric features from a single CV JSON."""
    with cv_path.open("r", encoding="utf-8") as f:
        cv = json.load(f)

    total_exp_months = sum(
        job.get("duration_months", 0) for job in cv.get("professional_experience", [])
    )
    n_jobs = len(cv.get("professional_experience", []))
    n_degrees = len(cv.get("education_background", []))

    skills = cv.get("skills", {}) or {}
    n_skills = 0
    for v in skills.values():
        if isinstance(v, list):
            n_skills += len(v)

    return {
        "total_exp_months": float(total_exp_months),
        "n_jobs": float(n_jobs),
        "n_degrees": float(n_degrees),
        "n_skills": float(n_skills),
    }


def build_feature_table(
    metadata_df: pd.DataFrame,
    cv_dir: Path,
    filename_col: str = "filename",
) -> pd.DataFrame:
    """Merge metadata with extracted numeric features from JSON CVs in `cv_dir`."""
    features: List[Dict[str, float]] = []
    missing: List[str] = []

    for fn in metadata_df[filename_col].astype(str).tolist():
        path = cv_dir / fn
        if not path.exists():
            missing.append(fn)
            continue
        row = {filename_col: fn}
        row.update(load_cv_features(path))
        features.append(row)

    if missing:
        print(
            f"[build_feature_table] WARNING: {len(missing)} JSON files not found. "
            f"Example: {missing[:5]}"
        )

    feat_df = pd.DataFrame(features)
    merged = metadata_df.merge(feat_df, on=filename_col, how="left")
    return merged


def _global_max_normalize(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        maxv = pd.to_numeric(out[col], errors="coerce").max()
        if pd.isna(maxv) or maxv <= 0:
            out[f"{col}_norm"] = 0.0
        else:
            out[f"{col}_norm"] = (
                pd.to_numeric(out[col], errors="coerce").fillna(0.0) / float(maxv)
            )
    return out


def _percentile_normalize(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Map each feature to [0,1] via percentile rank (robust to outliers)."""
    out = df.copy()
    for col in cols:
        x = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        # rank(pct=True) gives (1..n)/n; map to [0,1]
        out[f"{col}_norm"] = x.rank(pct=True, method="average").astype(float)
    return out


def compute_score(
    df: pd.DataFrame,
    feature_cols: Sequence[str] = tuple(DEFAULT_FEATURE_COLS),
    weights: Optional[Dict[str, float]] = None,
    normalize: str = "global_max",
    score_col: str = "score",
    noise_std: float = 0.0,
    seed: int = 12229942,
    bias_cfg: Optional[BiasConfig] = None,
) -> pd.DataFrame:
    """Compute a qualification score.

    Parameters
    ----------
    normalize:
      - "global_max": divide each feature by its global max (matches your notebook)
      - "percentile": percentile-rank normalization (often more robust)
      - "none": raw values (not recommended unless you re-tune weights)

    noise_std:
      Additive Gaussian noise applied *after* normalization and weighting.
      This reduces deterministic ties and makes the synthetic task more realistic.

    bias_cfg:
      Optional controlled label bias, applied after noise.
    """
    if weights is None:
        weights = {
            "total_exp_months": 0.60,
            "n_degrees": 0.20,
            "n_skills": 0.15,
            "n_jobs": 0.05,
        }

    for c in feature_cols:
        if c not in weights:
            raise ValueError(f"Missing weight for feature '{c}'. Provide it in `weights`.")

    work = df.copy()

    if normalize == "global_max":
        work = _global_max_normalize(work, feature_cols)
        score = 0.0
        for c in feature_cols:
            score = score + float(weights[c]) * work[f"{c}_norm"].astype(float)
        work[score_col] = score
    elif normalize == "percentile":
        work = _percentile_normalize(work, feature_cols)
        score = 0.0
        for c in feature_cols:
            score = score + float(weights[c]) * work[f"{c}_norm"].astype(float)
        work[score_col] = score
    elif normalize == "none":
        score = 0.0
        for c in feature_cols:
            score = score + float(weights[c]) * pd.to_numeric(work[c], errors="coerce").fillna(0.0)
        work[score_col] = score
    else:
        raise ValueError(f"Unknown normalize='{normalize}'.")

    if noise_std and noise_std > 0:
        rng = np.random.default_rng(seed)
        work[score_col] = work[score_col].astype(float) + rng.normal(0.0, float(noise_std), size=len(work))

    if bias_cfg is not None:
        m = bias_cfg.mask(work)
        work.loc[m, score_col] = work.loc[m, score_col] - float(bias_cfg.penalty)

    # keep scores in a sensible range for interpretability
    work[score_col] = work[score_col].astype(float).clip(lower=0.0)

    return work


def label_top_fraction_within_groups(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    score_col: str = "score",
    selectivity: float = 0.30,
    label_col: str = "label",
) -> pd.DataFrame:
    """Assign label=1 to the top `selectivity` fraction within each group.

    Tie handling:
      - threshold by quantile
      - if ties collapse everything, enforce top-k by sorting
    """
    if not (0.0 < selectivity < 1.0):
        raise ValueError("selectivity must be in (0, 1).")

    q = 1.0 - float(selectivity)
    out = df.copy()

    def _label_one_group(scores: pd.Series) -> pd.Series:
        n = len(scores)
        if n <= 1:
            return pd.Series([0] * n, index=scores.index)

        thr = scores.quantile(q)
        lab = (scores >= thr).astype(int)

        if lab.nunique() == 1:
            k = int(np.ceil(selectivity * n))
            k = min(max(k, 1), n - 1)  # keep at least 1 pos and 1 neg when possible
            idx_sorted = scores.sort_values().index
            lab[:] = 0
            lab.loc[idx_sorted[-k:]] = 1
        return lab

    out[label_col] = out.groupby(list(group_cols))[score_col].transform(_label_one_group).astype(int)
    return out


def build_labeled_metadata(
    metadata_df: pd.DataFrame,
    cv_dir: Path,
    group_cols: Sequence[str] = ("sector", "years_professional_experience"),
    selectivity: float = 0.30,
    normalize: str = "global_max",
    noise_std: float = 0.02,
    seed: int = 12229942,
) -> pd.DataFrame:
    """Build metadata with extracted features and *unbiased* labels.

    Outputs columns:
      - score_unbiased
      - label_unbiased
    """
    df = build_feature_table(metadata_df, cv_dir=cv_dir)
    df = compute_score(df, normalize=normalize, score_col="score_unbiased", noise_std=noise_std, seed=seed, bias_cfg=None)
    df = label_top_fraction_within_groups(df, group_cols=group_cols, score_col="score_unbiased", selectivity=selectivity, label_col="label_unbiased")
    return df


def add_biased_labels(
    df: pd.DataFrame,
    bias_cfg: BiasConfig,
    group_cols: Sequence[str] = ("sector", "years_professional_experience"),
    selectivity: float = 0.30,
    normalize: str = "global_max",
    noise_std: float = 0.02,
    seed: int = 12229942,
) -> pd.DataFrame:
    """Add an additional *biased* score/label pair to an already-featured dataframe.

    Adds columns:
      - score_biased_<bias_cfg.name>
      - label_biased_<bias_cfg.name>
    """
    out = df.copy()
    score_col = f"score_biased_{bias_cfg.name}"
    label_col = f"label_biased_{bias_cfg.name}"
    out = compute_score(out, normalize=normalize, score_col=score_col, noise_std=noise_std, seed=seed, bias_cfg=bias_cfg)
    out = label_top_fraction_within_groups(out, group_cols=group_cols, score_col=score_col, selectivity=selectivity, label_col=label_col)
    return out


def label_diagnostics(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("sector", "years_professional_experience"),
    label_col: str = "label_unbiased",
) -> pd.DataFrame:
    """Per-group counts and shortlist rates for quick sanity checks."""
    diag = (
        df.groupby(list(group_cols))
        .agg(total_candidates=(label_col, "count"), shortlisted=(label_col, "sum"))
        .reset_index()
    )
    diag["shortlist_rate"] = diag["shortlisted"] / diag["total_candidates"]
    return diag


def group_selection_rates(
    df: pd.DataFrame,
    label_col: str,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Selection/shortlist rates by demographic group(s) for reporting."""
    out = (
        df.groupby(list(group_cols))
        .agg(n=(label_col, "count"), shortlisted=(label_col, "sum"))
        .reset_index()
    )
    out["shortlist_rate"] = out["shortlisted"] / out["n"]
    return out
