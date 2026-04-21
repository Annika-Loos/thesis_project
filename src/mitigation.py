from __future__ import annotations
import numpy as np
import pandas as pd

def compute_reweighing_weights(
    df: pd.DataFrame,
    label_col: str,
    target_attr: str = "gender",
    attr_col: str = "demographic_var",
    val_col: str = "demographic_var_value",
) -> np.ndarray:
    """
    Compute sample reweighing weights for a single sensitive attribute.
    
    Returns a weight array of length len(df). Rows NOT belonging to
    target_attr get weight 1.0 (unchanged). Rows belonging to target_attr
    get Kamiran & Calders reweighing weights.
    """
    # Start with all weights = 1
    weights = np.ones(len(df), dtype=float)

    # Get a boolean mask for the target attribute rows
    mask = df[attr_col].astype(str) == target_attr
    sub = df[mask].copy()

    if sub.empty:
        return weights

    sub["A"] = sub[val_col].astype(str)
    sub["Y"] = sub[label_col].astype(int)

    n = len(sub)  # total rows belonging to this attribute

    p_a = sub["A"].value_counts(normalize=True).to_dict()
    p_y = sub["Y"].value_counts(normalize=True).to_dict()
    # Joint probability within this attribute's subset
    p_ay = (sub.groupby(["A", "Y"]).size() / n).to_dict()

    w_map = {}
    for (a, y), p in p_ay.items():
        if p > 0:
            w_map[(a, y)] = (p_a[a] * p_y[y]) / p

    df_reset = df.reset_index(drop=True) 
    sub_reset_idx = df_reset[df_reset[attr_col].astype(str) == target_attr].index

    for pos in sub_reset_idx:
        row = df_reset.loc[pos]
        a = str(row[val_col])
        y = int(row[label_col])
        weights[pos] = w_map.get((a, y), 1.0)

    return weights


def apply_massaging(
    df: pd.DataFrame,
    label_col: str,
    target_attr: str = "gender",
    privileged_val: str = "Man",
    unprivileged_val: str = "Woman",
    attr_col: str = "demographic_var",
    val_col: str = "demographic_var_value",
    score_col: str = "score_unbiased",
) -> pd.DataFrame:
    """
    Kamiran & Calders massaging: relabel borderline instances to reduce
    the correlation between the sensitive attribute and the target label.
    
    - Finds privileged positive instances (Man, y=1) ranked lowest by score
      and flips them to y=0.
    - Finds unprivileged negative instances (Woman, y=0) ranked highest by score
      and flips them to y=1.
    - The number of flips is determined by the discrimination score
      (difference in positive rates between groups).
    """
    df = df.reset_index(drop=True).copy()

    mask_attr = df[attr_col].astype(str) == target_attr
    sub = df[mask_attr]

    if sub.empty:
        return df

    if score_col not in df.columns:
        raise ValueError(
            f"score_col='{score_col}' not found in dataframe. "
            f"Available columns: {df.columns.tolist()}"
        )

    priv   = sub[sub[val_col].astype(str) == privileged_val]
    unpriv = sub[sub[val_col].astype(str) == unprivileged_val]

    if priv.empty or unpriv.empty:
        return df

    # Discrimination = P(y=1 | privileged) - P(y=1 | unprivileged)
    p_priv   = float(priv[label_col].astype(int).mean())
    p_unpriv = float(unpriv[label_col].astype(int).mean())
    disc = p_priv - p_unpriv

    if disc <= 0:
        # No discrimination to correct
        print(f"[massaging] No discrimination detected (disc={disc:.4f}). No labels changed.")
        return df

    # Number of flips needed (proportional to discrimination * group size)
    n_flip = max(1, int(round(disc * len(sub) / 2)))
    print(f"[massaging] disc={disc:.4f}, flipping {n_flip} labels per group")

    # Privileged positives with LOWEST score → most borderline → flip to 0
    priv_pos = df[(mask_attr) &
                  (df[val_col].astype(str) == privileged_val) &
                  (df[label_col].astype(int) == 1)].copy()
    priv_pos_sorted = priv_pos.sort_values(score_col, ascending=True)
    flip_priv_idx = priv_pos_sorted.index[:n_flip]

    # Unprivileged negatives with HIGHEST score → most borderline → flip to 1
    unpriv_neg = df[(mask_attr) &
                    (df[val_col].astype(str) == unprivileged_val) &
                    (df[label_col].astype(int) == 0)].copy()
    unpriv_neg_sorted = unpriv_neg.sort_values(score_col, ascending=False)
    flip_unpriv_idx = unpriv_neg_sorted.index[:n_flip]

    df.loc[flip_priv_idx,   label_col] = 0
    df.loc[flip_unpriv_idx, label_col] = 1

    print(f"[massaging] Flipped {len(flip_priv_idx)} privileged 1→0, "
          f"{len(flip_unpriv_idx)} unprivileged 0→1")

    return df