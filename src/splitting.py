"""
Dataset splitting utilities.

Default target split: train/val/test = 70/10/20 (user request).

Stratification:
- Prefer strong stratification on (sector, years_professional_experience, label_unbiased)
- If that fails (tiny groups), fall back to (sector, label_unbiased) then to label only.

This makes downstream fairness analysis more stable because val/test don't drift too much
across sector/experience.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.10
    test_frac: float = 0.20
    seed: int = 12229942

    def __post_init__(self):
        s = self.train_frac + self.val_frac + self.test_frac
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1. Got {s}.")


def _make_strat_key(df: pd.DataFrame, cols: Sequence[str], sep: str = "|") -> pd.Series:
    return df[list(cols)].astype(str).agg(sep.join, axis=1)


def split_train_val_test(
    df: pd.DataFrame,
    cfg: SplitConfig = SplitConfig(),
    filename_col: str = "filename",
    sector_col: str = "sector",
    exp_col: str = "years_professional_experience",
    label_col: str = "label_unbiased",
) -> pd.DataFrame:
    """
    Returns a dataframe with columns: [filename, split]
    """
    work = df.copy()

    # Build candidate strat keys (strong -> weak)
    key_full = _make_strat_key(work, [sector_col, exp_col, label_col])
    key_sector_label = _make_strat_key(work, [sector_col, label_col])
    key_label = work[label_col].astype(str)

    # First split: train vs temp (val+test)
    temp_frac = cfg.val_frac + cfg.test_frac  # here: 0.30
    strat_series = None
    for candidate in (key_full, key_sector_label, key_label):
        try:
            train_df, temp_df = train_test_split(
                work,
                test_size=temp_frac,
                random_state=cfg.seed,
                stratify=candidate,
            )
            strat_series = candidate
            break
        except ValueError:
            continue

    if strat_series is None:
        # Last resort: no stratification
        train_df, temp_df = train_test_split(
            work,
            test_size=temp_frac,
            random_state=cfg.seed,
            stratify=None,
        )

    # Second split: temp -> val/test.
    # Want test_frac relative to temp.
    test_rel = cfg.test_frac / temp_frac  # 0.20 / 0.30 = 2/3
    # Recompute keys for temp
    key_full_t = _make_strat_key(temp_df, [sector_col, exp_col, label_col])
    key_sector_label_t = _make_strat_key(temp_df, [sector_col, label_col])
    key_label_t = temp_df[label_col].astype(str)

    strat2 = None
    for candidate in (key_full_t, key_sector_label_t, key_label_t):
        try:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=test_rel,
                random_state=cfg.seed,
                stratify=candidate,
            )
            strat2 = candidate
            break
        except ValueError:
            continue

    if strat2 is None:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_rel,
            random_state=cfg.seed,
            stratify=None,
        )

    out = pd.concat(
        [
            train_df[[filename_col]].assign(split="train"),
            val_df[[filename_col]].assign(split="val"),
            test_df[[filename_col]].assign(split="test"),
        ],
        axis=0,
        ignore_index=True,
    )

    # safety
    if out[filename_col].duplicated().any():
        raise RuntimeError("Duplicate filenames across splits detected.")

    return out
