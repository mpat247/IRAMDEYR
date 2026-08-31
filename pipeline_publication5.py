#!/usr/bin/env python3
"""
Publication-ready end-to-end pipeline for:
Predicting Past-Year Major Depressive Episode (IRAMDEYR) using NSDUH 2023

Key design choices in this version
----------------------------------
1. Feature selection is skipped entirely; existing Tier 1 / 2 / 3 feature sets are used.
2. SVM is removed entirely.
3. All figures are saved as SVG by default; optional PDP panels also save as SVG.
4. Combined ROC / PR / calibration / decision-curve figures are generated per tier
   with all models overlaid and CV mean ± SD shading.
5. Threshold selection is done on TRAINING DATA ONLY using out-of-fold probabilities.
   The test set is used once, only for final locked evaluation.
6. Baseline table includes weighted descriptive summaries and approximate weighted
   association p-values based on weighted GLM / weighted LR tests (with robust SEs).
7. Calibration metrics are added:
   - Brier score
   - Calibration intercept
   - Calibration slope
8. Bootstrap CIs are added for final operating-point metrics (fully weighted):
   - Accuracy, Sensitivity, Specificity, PPV, NPV, AUC, AP, F1, F2, Brier
9. A primary summary table is generated with one locked result row per model per tier.

Example
-------
python pipeline_publication_pubready_v2.py \
    --raw-data data/raw/NSDUH_2023_Tab.txt \
    --run-all --run-shap
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Optional dependencies
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from scipy.stats import randint as sp_randint, uniform as sp_uniform
    from scipy.stats import chi2
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# ---------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TARGET = "IRAMDEYR"
WEIGHT = "ANALWT2_C"

SEARCH_CV_FOLDS = 5
REPORT_CV_FOLDS = 5
REPORT_CV_REPEATS = 5
N_BOOT = 1000
CALIBRATION_BINS = 10
PDP_JITTER = 0.08
PDP_CONT_GRID = 25
PDP_CONT_BOOT = 200

# Existing tier lists from prior feature-selection work
TIER_1 = [
    "CATAG3",
    "IRSEX",
    "IRMARIT_ABS",
    "IRPINC3_ABS",
    "SVYRDUDANY",
    "WRKDRGHLP_ABS",
]
TIER_2 = TIER_1 + [
    "IRSUICTHNK",
    "KSSLR6MAX_ABS",
    "RCVYMHPRB",
    "IRIMPGOUT_ABS",
]
TIER_3 = TIER_2 + [
    "ILLEMFLAG",
    "WHODASDASC_ABS",
    "MJCMOTHYR",
    "MJSMKYR",
    "ILLEMMON",
    "COCLALCUSE_ABS",
    "DIFOBTCRK",
    "MHNTENFCV",
    "MHNTINSCV",
]
TIERS = {
    "Tier_1_Basic": TIER_1,
    "Tier_2_Clinical": TIER_2,
    "Tier_3_Personalized": TIER_3,
}


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
@dataclass
class Paths:
    project_root: Path
    raw_data: Path
    data_dir: Path
    cleaned_dir: Path
    prepared_dir: Path
    results_dir: Path
    figures_dir: Path
    models_dir: Path
    tiers_dir: Path

    @property
    def cleaned_pkl(self) -> Path:
        return self.cleaned_dir / "NSDUH_2023_clean.pkl"

    @property
    def prepared_pkl(self) -> Path:
        return self.prepared_dir / "NSDUH_2023_prepared.pkl"


def make_paths(project_root: Path, raw_data: Path) -> Paths:
    data_dir = project_root / "data"
    cleaned_dir = data_dir / "cleaned"
    prepared_dir = data_dir / "prepared"
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    models_dir = project_root / "models"
    tiers_dir = project_root / "tiers"

    for p in [data_dir, cleaned_dir, prepared_dir, results_dir, figures_dir, models_dir, tiers_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return Paths(
        project_root=project_root,
        raw_data=raw_data,
        data_dir=data_dir,
        cleaned_dir=cleaned_dir,
        prepared_dir=prepared_dir,
        results_dir=results_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        tiers_dir=tiers_dir,
    )


# ---------------------------------------------------------------------
# Plot config
# ---------------------------------------------------------------------
def configure_publication_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.format": "svg",
            "svg.fonttype": "path",
        }
    )


def export_tier_csvs(paths: Paths) -> None:
    for tier_name, feats in TIERS.items():
        out = paths.tiers_dir / f"{tier_name.lower()}_features.csv"
        pd.DataFrame({"feature": feats}).to_csv(out, index=False)

FEATURE_LABELS = {
    "CATAG3": "Age group",
    "IRSEX": "Sex",
    "IRMARIT_ABS": "Marital status",
    "IRPINC3_ABS": "Family income",
    "SVYRDUDANY": "Any drug use disorder severity",
    "WRKDRGHLP_ABS": "Workplace assistance access",
    "IRSUICTHNK": "Serious thoughts of suicide",
    "KSSLR6MAX_ABS": "Psychological distress severity",
    "RCVYMHPRB": "Received mental health treatment/counseling",
    "IRIMPGOUT_ABS": "Difficulty going out alone",
    "ILLEMFLAG": "Any illegal drug use",
    "WHODASDASC_ABS": "WHODAS disability score",
    "MJCMOTHYR": "Marijuana use (past year)",
    "MJSMKYR": "Marijuana smoking (past year)",
    "ILLEMMON": "Illegal drug use other than Marijuana (past month)",
    "COCLALCUSE_ABS": "Alcohol use frequency because of COVID",
    "DIFOBTCRK": "Difficulty obtaining crack",
    "MHNTENFCV": "Mental health treatment not covered by employer",
    "MHNTINSCV": "Mental health treatment not covered by insurance",
}

LEVEL_LABELS = {
    "CATAG3": {
        2: "Age 18–25",
        3: "Age 26–34",
        4: "Age 35–49",
        5: "Age 50+",
        "2": "Age 18–25",
        "3": "Age 26–34",
        "4": "Age 35–49",
        "5": "Age 50+",
    },
    "IRSEX": {
        1: "Male",
        2: "Female",
        "1": "Male",
        "2": "Female",
    },
    "SVYRDUDANY": {
        4: "No disorder",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        "4": "No disorder",
        "1": "Mild",
        "2": "Moderate",
        "3": "Severe",
    },
}
# ---------------------------------------------------------------------
# Data cleaning and preparation
# ---------------------------------------------------------------------
def clean_nsduh(raw_path: Path, cleaned_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, sep="\t", low_memory=False)
    df = df[df["AGE3"] >= 4].copy()

    nan_codes = [
        94, 97, 98, 85,
        994, 997, 998, 985,
        9994, 9997, 9998, 9985,
        99, 999, 9999,
        89, 989, 9989,
    ]
    df = df.replace(nan_codes, np.nan)
    df = df.dropna(subset=[TARGET]).copy()

    drop_patterns = [
        r"^CASEID", r"^QUESTID", r"^QUESTID2", r"^PANEL", r"^VERSION",
        r"^INTV", r"^FILE", r"^YEAR",
        r"^WT", r"WGT", r"WEIGHT", r"ANALWT(?!2_C$)",
        r"^VESTR", r"^VEREP", r"^ESTRAT",
        r"_A$", r"_E$", r"_ORIG$", r"_R$", r"_RC$",
    ]
    cols_to_drop = [c for c in df.columns if any(re.search(pattern, c) for pattern in drop_patterns)]
    cols_to_drop = [c for c in cols_to_drop if c != WEIGHT]
    cols_to_drop += [c for c in ["WTANSWER", "WTPOUND2", "FILEDATE"] if c in df.columns]
    cols_to_drop = sorted(set(cols_to_drop))
    df = df.drop(columns=cols_to_drop, errors="ignore")

    na_pct = df.isna().mean() * 100
    cols_over_99 = na_pct[na_pct > 99].index.tolist()
    df = df.drop(columns=cols_over_99, errors="ignore")

    leakage_vars = [
        "IIAMDEYR", "AMDEYR", "AMDELT", "IRAMDELT", "IIAMDELT",
        "AMDEIMP", "IRAMDEIMP", "IIAMDEIMP", "AMDETXRX",
        "ADSMMDEA",
        "D_MDEA1", "D_MDEA2", "D_MDEA3", "D_MDEA4", "D_MDEA5",
        "D_MDEA6", "D_MDEA7", "D_MDEA8", "D_MDEA9",
        "AD_MDEA1", "AD_MDEA2", "AD_MDEA3", "AD_MDEA4",
        "AD_MDEA5", "AD_MDEA6", "AD_MDEA7", "AD_MDEA8",
        "AD_MDEA11", "AD_MDEA21", "AD_MDEA31", "AD_MDEA41",
        "AD_MDEA51", "AD_MDEA61", "AD_MDEA71", "AD_MDEA81", "AD_MDEA91",
        "ADPB2WK",
        "ADDPREV", "ADDSCEV", "ADLOSEV", "ADLSI2WK", "ADDPR2WK",
        "ADWRHRS", "ADWRDST", "ADWRCHR", "ADWRIMP", "ADDPPROB",
        "ARXMDEYR", "ATXMDEYR", "AHLTMDE",
    ]
    df = df.drop(columns=[c for c in leakage_vars if c in df.columns], errors="ignore")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    df.to_pickle(cleaned_path)
    return df


def _weighted_mean(x: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    x = np.asarray(x, dtype=float)
    if w is None:
        return float(np.mean(x))
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w)
    if not np.any(mask):
        return np.nan
    denom = np.sum(w[mask])
    if denom <= 0:
        return np.nan
    return float(np.sum(x[mask] * w[mask]) / denom)


def _series_is_continuous_numeric(s: pd.Series, min_unique: int = 8) -> bool:
    # Explicitly exclude strings and categoricals from being treated as continuous
    if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s):
        return False
    s_num = pd.to_numeric(s, errors="coerce")
    return s_num.notna().sum() >= 20 and s_num.nunique(dropna=True) >= min_unique


def _series_is_discrete(s: pd.Series, max_unique: int = 20) -> bool:
    s_nonmissing = s.dropna()
    if s_nonmissing.empty:
        return False
    if isinstance(s_nonmissing.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s_nonmissing):
        return s_nonmissing.nunique(dropna=True) <= max_unique
    s_num = pd.to_numeric(s_nonmissing, errors="coerce")
    return s_num.notna().sum() == len(s_nonmissing) and s_num.nunique(dropna=True) <= max_unique


def _ordered_categories_for_plot(s: pd.Series) -> List:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [c for c in s.cat.categories if pd.notna(c)]
    vals = s.dropna().unique().tolist()
    try:
        return sorted(vals)
    except Exception:
        return vals


def _discrete_tick_label(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def plot_discrete_effect(
    out_svg: Path,
    model,
    X_ref: pd.DataFrame,
    feature: str,
    weights: Optional[np.ndarray] = None,
    label_func=None,
) -> bool:
    s = X_ref[feature]
    categories = _ordered_categories_for_plot(s)
    if len(categories) < 2:
        return False

    X_base = X_ref.copy()
    p_base = model.predict_proba(X_base)[:, 1]

    effects_all = []
    x_all = []
    freqs = []
    mean_effects = [] # Added to store the actual PDP average

    for i, cat in enumerate(categories):
        X_tmp = X_ref.copy()
        if isinstance(X_tmp[feature].dtype, pd.CategoricalDtype):
            X_tmp[feature] = pd.Categorical([cat] * len(X_tmp), categories=X_tmp[feature].cat.categories, ordered=X_tmp[feature].cat.ordered)
        else:
            X_tmp[feature] = cat
            
        p_cf = model.predict_proba(X_tmp)[:, 1]
        eff = p_cf - p_base
        
        effects_all.append(eff)
        x_all.append(np.full(len(eff), i, dtype=float))
        
        if weights is None:
            freqs.append(float((s == cat).mean()))
            mean_effects.append(float(np.mean(eff)))
        else:
            mask = (s == cat).to_numpy()
            freqs.append(_weighted_mean(mask.astype(float), weights))
            mean_effects.append(_weighted_mean(eff, weights))

    effects_all = np.concatenate(effects_all)
    x_all = np.concatenate(x_all)
    freqs = np.asarray(freqs, dtype=float)

    jitter = np.random.normal(0, PDP_JITTER, size=len(x_all))
    x_scatter = x_all + jitter

    fig, ax1 = plt.subplots(figsize=(7.6, 5.0))
    ax1.bar(np.arange(len(categories)), freqs, edgecolor="black", alpha=0.25, color="lightgray")
    ax1.set_ylabel("Category frequency" if weights is None else "Weighted category frequency")
    ax1.set_xticks(np.arange(len(categories)))
    ax1.set_xticklabels([_discrete_tick_label(c) for c in categories], rotation=45, ha="right")
    ax1.set_xlabel(label_func(feature) if label_func else feature)

    ax2 = ax1.twinx()
    # Plot ICE (individual points) faintly
    ax2.scatter(x_scatter, effects_all, s=10, alpha=0.15, color="steelblue", zorder=1)
    
    # Plot actual PDP (mean effect line) prominently
    ax2.plot(np.arange(len(categories)), mean_effects, color="darkred", marker="o", markersize=8, linewidth=2.5, zorder=2, label="Mean Marginal Effect")
    
    ax2.axhline(0, linestyle="--", linewidth=1, color="black")
    ax2.set_ylabel("Δ Predicted Risk")
    ax2.legend(loc="upper left", frameon=False)

    plt.title(f"Partial Dependence (Discrete): {label_func(feature) if label_func else feature}")
    plt.tight_layout()
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()
    return True


def plot_continuous_effect(
    out_svg: Path,
    model,
    X_ref: pd.DataFrame,
    weights: Optional[np.ndarray],
    feature: str,
    grid_points: int = PDP_CONT_GRID,
    boot_iters: int = PDP_CONT_BOOT,
    label_func=None,
) -> bool:
    x = pd.to_numeric(X_ref[feature], errors="coerce").values
    mask = np.isfinite(x)
    if mask.sum() < 20:
        return False

    X_ref2 = X_ref.loc[mask].copy()
    x = x[mask]
    w_ref = None if weights is None else np.asarray(weights, dtype=float)[mask]
    orig_dtype = X_ref2[feature].dtype

    qs = np.linspace(0.05, 0.95, grid_points)
    grid = np.unique(np.quantile(x, qs))
    if len(grid) < 2:
        return False

    n_samples = len(X_ref2)
    pred_matrix = np.zeros((n_samples, len(grid)))
    
    for j, v in enumerate(grid):
        X_tmp = X_ref2.copy()
        X_tmp[feature] = float(v)
        try:
            X_tmp[feature] = X_tmp[feature].astype(orig_dtype)
        except:
            pass
        pred_matrix[:, j] = model.predict_proba(X_tmp)[:, 1]

    mean = np.array([_weighted_mean(pred_matrix[:, j], w_ref) for j in range(len(grid))])

    # Create figure explicitly
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    
    ax.plot(grid, mean, linewidth=2.5, color="darkred", label="Mean Marginal Effect")
    
    if boot_iters and boot_iters > 1:
        ys = []
        for i in range(boot_iters):
            idx = resample(np.arange(n_samples), replace=True, random_state=42 + i)
            wb = None if w_ref is None else w_ref[idx]
            y_boot = np.array([_weighted_mean(pred_matrix[idx, j], wb) for j in range(len(grid))])
            ys.append(y_boot)
            
        ys = np.asarray(ys, dtype=float)
        lo = np.percentile(ys, 2.5, axis=0)
        hi = np.percentile(ys, 97.5, axis=0)
        ax.fill_between(grid, lo, hi, color="darkred", alpha=0.2, label="95% CI")
        
    ax.set_xlabel(label_func(feature) if label_func else feature)
    ax.set_ylabel("Predicted Risk" + (" (Weighted)" if weights is not None else ""))
    ax.set_title(f"Partial Dependence: {label_func(feature) if label_func else feature}")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, alpha=0.25)
    
    # Critical Fix: Use fig.savefig instead of plt.savefig
    plt.tight_layout()
    fig.savefig(out_svg, bbox_inches="tight", facecolor='white')
    plt.close(fig) # Close specific figure
    return True


def run_partial_dependence_plots(
    df: pd.DataFrame,
    tier_name: str,
    paths: Paths,
    features_to_plot: List[str],
    model_name: str = "RF",
) -> None:
    print(f"\n[PDP] Running PDP-like plots for {model_name} / {tier_name}")

    prefix_map = {
        "RF": "rf",
        "XGB": "xgb",
        "LGBM": "lightgbm",
    }
    if model_name not in prefix_map:
        raise ValueError(f"Unsupported model for PDP-like plots: {model_name}")
    if tier_name not in TIERS:
        raise ValueError(f"Unknown tier: {tier_name}")

    model_path = paths.models_dir / f"{prefix_map[model_name]}_{tier_name.lower()}.joblib"
    if not model_path.exists():
        print(f"[WARN] Model not found: {model_path}")
        return

    tier_features = [c for c in TIERS[tier_name] if c in df.columns]
    requested_features = [f for f in features_to_plot if f in tier_features]
    if not requested_features:
        print(f"[WARN] No requested PDP features are present in {tier_name}.")
        return

    missing_cols = [c for c in TIERS[tier_name] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Prepared data is missing required tier columns: {missing_cols}")

    model = joblib.load(model_path)

    X_ref = df[tier_features].copy()
    weights = None
    if WEIGHT in df.columns:
        weights = pd.to_numeric(df[WEIGHT], errors="coerce").fillna(1.0).astype(float).values

    out_dir = paths.figures_dir / "pdp_like" / model_name / tier_name
    out_dir.mkdir(parents=True, exist_ok=True)

    made_any = False
    for feature in requested_features:
        s = X_ref[feature]
        out_svg = out_dir / f"pdp_like_{model_name.lower()}_{tier_name.lower()}_{feature}.svg"

        try:
            ok = False
            if _series_is_continuous_numeric(s):
                ok = plot_continuous_effect(
                    out_svg=out_svg,
                    model=model,
                    X_ref=X_ref,
                    weights=weights,
                    feature=feature,
                    grid_points=PDP_CONT_GRID,
                    boot_iters=PDP_CONT_BOOT,
                    label_func=pretty_feature_name,
                )
            elif _series_is_discrete(s):
                ok = plot_discrete_effect(
                    out_svg=out_svg,
                    model=model,
                    X_ref=X_ref,
                    feature=feature,
                    weights=weights,
                    label_func=pretty_feature_name,
                )
            else:
                # fallback: try treating anything else as a categorical/discrete variable
                ok = plot_discrete_effect(
                    out_svg=out_svg,
                    model=model,
                    X_ref=X_ref,
                    feature=feature,
                    weights=weights,
                    label_func=pretty_feature_name,
                )

            if ok:
                made_any = True
                print(f"[PDP] Saved: {out_svg}")
            else:
                print(f"[WARN] Could not create PDP-like plot for '{feature}' in {tier_name}")
        except Exception as e:
            print(f"[WARN] Skipping PDP-like for feature '{feature}' in {tier_name}: {e}")

    if not made_any:
        print(f"[WARN] No PDP-like plots were generated for {model_name} / {tier_name}")

def prepare_features(cleaned_df: pd.DataFrame, prepared_path: Path) -> pd.DataFrame:
    df = cleaned_df.copy()

    features_to_remove = ["ADRX12MO", "IRMHTRXMED", "ADWRENRG", "ADWRSLNO", "ADPSHMGT"]
    df = df.drop(columns=[c for c in features_to_remove if c in df.columns], errors="ignore")

    raw_to_recode = {
        "AGE3": "CATAG3",
        "CAMHRCVR": "RCVYMHPRB",
        "MJSKNYR": "MJCMOTHYR",
        "DIFGETCRK": "DIFOBTCRK",
        "MHTUNENFCV": "MHNTENFCV",
        "MHTUNINSCV": "MHNTINSCV",
    }
    for raw_col, derived_col in raw_to_recode.items():
        if derived_col not in df.columns and raw_col in df.columns:
            df[derived_col] = df[raw_col]
        if raw_col in df.columns and raw_col != derived_col:
            df = df.drop(columns=[raw_col], errors="ignore")

    if "IRMARIT" in df.columns:
        map_ = {1: "Married", 2: "Widowed-Divorced-or-Separated", 3: "Widowed-Divorced-or-Separated", 4: "Never Married"}
        order = ["Married", "Widowed-Divorced-or-Separated", "Never Married"]
        df["IRMARIT_ABS"] = pd.Categorical(df["IRMARIT"].map(map_), categories=order, ordered=True)
        df = df.drop(columns=["IRMARIT"])

    if "IRPINC3" in df.columns:
        map_ = {
            1: "a_<10k",
            2: "b_10k-to-40k",
            3: "b_10k-to-40k",
            4: "b_10k-to-40k",
            5: "c_50k-to-75k",
            6: "c_50k-to-75k",
            7: "d_75k+",
        }
        order = ["a_<10k", "b_10k-to-40k", "c_50k-to-75k", "d_75k+"]
        df["IRPINC3_ABS"] = pd.Categorical(df["IRPINC3"].map(map_), categories=order, ordered=True)
        df = df.drop(columns=["IRPINC3"])

    if "WRKDRGHLP" in df.columns:
        map_ = {
            1: "Having access to assistance program at work",
            2: "Not having access to assistance program at work",
        }
        order = [
            "Having access to assistance program at work",
            "Not having access to assistance program at work",
            "No need or do not know about assistance program at work",
        ]
        df["WRKDRGHLP_ABS"] = df["WRKDRGHLP"].map(map_).fillna("No need or do not know about assistance program at work")
        df["WRKDRGHLP_ABS"] = pd.Categorical(df["WRKDRGHLP_ABS"], categories=order, ordered=True)
        df = df.drop(columns=["WRKDRGHLP"])

    if "KSSLR6MAX" in df.columns:
        bins = [-1, 6, 12, 18, 24]
        labels = ["0-6", "07-12", "13-18", "19-24"]
        df["KSSLR6MAX_ABS"] = pd.cut(df["KSSLR6MAX"], bins=bins, labels=labels)
        df["KSSLR6MAX_ABS"] = pd.Categorical(df["KSSLR6MAX_ABS"], categories=labels, ordered=True)
        df = df.drop(columns=["KSSLR6MAX"])

    if "IRIMPGOUT" in df.columns:
        map_ = {
            1: "b_No difficulty",
            2: "c_Mild difficulty",
            3: "d_Moderate difficulty",
            4: "e_Severe difficulty",
            5: "a_Not-Going-Out-Alone-Or-No-Difficulty",
            99: "a_Not-Going-Out-Alone-Or-No-Difficulty",
        }
        order = [
            "a_Not-Going-Out-Alone-Or-No-Difficulty",
            "b_No difficulty",
            "c_Mild difficulty",
            "d_Moderate difficulty",
            "e_Severe difficulty",
        ]
        df["IRIMPGOUT_ABS"] = df["IRIMPGOUT"].map(map_).fillna("a_Not-Going-Out-Alone-Or-No-Difficulty")
        df["IRIMPGOUT_ABS"] = pd.Categorical(df["IRIMPGOUT_ABS"], categories=order, ordered=True)
        df = df.drop(columns=["IRIMPGOUT"])

    if "WHODASDASC" in df.columns:
        map_ = {0: "0", 1: "1-2", 2: "1-2", 3: "3-4", 4: "3-4", 5: "5-6", 6: "5-6", 7: "7-8", 8: "7-8"}
        order = ["0", "1-2", "3-4", "5-6", "7-8"]
        df["WHODASDASC_ABS"] = pd.Categorical(df["WHODASDASC"].map(map_), categories=order, ordered=True)
        df = df.drop(columns=["WHODASDASC"])

    if "COCLALCUSE" in df.columns:
        raw_map = {
            1: "Drink much less or little less",
            2: "Drink about the same",
            3: "Drink a little more or much more",
        }
        label_map = {
            "Unknown/No PY Alcohol Use": "a_Unknown/No PY Alcohol Use",
            "Drink much less or little less": "b_Drink much less or little less",
            "Drink about the same": "c_Drink about the same",
            "Drink a little more or much more": "d_Drink a little more or much more",
        }
        order = [
            "a_Unknown/No PY Alcohol Use",
            "b_Drink much less or little less",
            "c_Drink about the same",
            "d_Drink a little more or much more",
        ]
        df["COCLALCUSE_ABS"] = df["COCLALCUSE"].map(raw_map).fillna("Unknown/No PY Alcohol Use")
        df["COCLALCUSE_ABS"] = pd.Categorical(df["COCLALCUSE_ABS"].map(label_map), categories=order, ordered=True)
        df = df.drop(columns=["COCLALCUSE"])

    df.to_pickle(prepared_path)
    return df

def pretty_feature_name(feature_name: str) -> str:
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]

    # one-hot encoded style names: BASE_LEVEL
    for base in sorted(FEATURE_LABELS.keys(), key=len, reverse=True):
        prefix = f"{base}_"
        if feature_name.startswith(prefix):
            raw_level = feature_name[len(prefix):]
            pretty_base = FEATURE_LABELS[base]
            pretty_level = LEVEL_LABELS.get(base, {}).get(raw_level, raw_level)
            return f"{pretty_base}: {pretty_level}"

    return feature_name


def pretty_code_meaning(var: str, level) -> str:
    return LEVEL_LABELS.get(var, {}).get(level, LEVEL_LABELS.get(var, {}).get(str(level), str(level)))

# ---------------------------------------------------------------------
# Baseline characteristics with approximate weighted p-values
# ---------------------------------------------------------------------
def weighted_mean_sd(series: pd.Series, weights: pd.Series) -> Tuple[float, float]:
    x = series.astype(float).to_numpy()
    w = weights.astype(float).to_numpy()
    mean = np.average(x, weights=w)
    var = np.average((x - mean) ** 2, weights=w)
    return float(mean), float(np.sqrt(var))


def weighted_percent(df: pd.DataFrame, var: str, target_value: Optional[int] = None) -> pd.Series:
    d = df.copy()
    if target_value is not None:
        d = d[d[TARGET] == target_value].copy()
    grp = d.groupby(var, dropna=False)[WEIGHT].sum()
    return (grp / grp.sum() * 100).round(1)


def format_p_value(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def approximate_weighted_p_continuous(df: pd.DataFrame, var: str) -> float:
    if not (HAS_STATSMODELS and HAS_SCIPY):
        return np.nan
    d = df[[TARGET, var, WEIGHT]].dropna().copy()
    if d.empty or d[var].nunique() < 2 or d[TARGET].nunique() < 2:
        return np.nan
    try:
        X = sm.add_constant(d[[var]].astype(float))
        y = d[TARGET].astype(float)
        model = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=d[WEIGHT].astype(float))
        # Adding HC1 robust standard errors
        fit = model.fit(cov_type='HC1')
        return float(fit.pvalues.get(var, np.nan))
    except Exception:
        return np.nan


def approximate_weighted_p_categorical(df: pd.DataFrame, var: str) -> float:
    if not (HAS_STATSMODELS and HAS_SCIPY):
        return np.nan
    d = df[[TARGET, var, WEIGHT]].dropna().copy()
    if d.empty or d[var].nunique() < 2 or d[TARGET].nunique() < 2:
        return np.nan
    try:
        X_full = pd.get_dummies(d[var].astype(str), drop_first=True)
        if X_full.shape[1] == 0:
            return np.nan
        X_full = sm.add_constant(X_full.astype(float), has_constant="add")
        X_null = sm.add_constant(pd.DataFrame(index=d.index), has_constant="add")
        y = d[TARGET].astype(float)
        w = d[WEIGHT].astype(float)

        # Adding HC1 robust standard errors
        fit_full = sm.GLM(y, X_full, family=sm.families.Binomial(), var_weights=w).fit(cov_type='HC1')
        fit_null = sm.GLM(y, X_null, family=sm.families.Binomial(), var_weights=w).fit(cov_type='HC1')

        lr = 2 * (fit_full.llf - fit_null.llf)
        df_diff = fit_full.df_model - fit_null.df_model
        p = 1 - chi2.cdf(lr, df_diff)
        return float(p)
    except Exception:
        return np.nan


def build_baseline_characteristics_table(paths: Paths, tier_name: str = "Tier_3_Personalized") -> pd.DataFrame:
    """
    Build a full baseline/descriptive table for all features in the requested tier.
    Uses the prepared dataset so the recoded publication features are included.
    """

    df = pd.read_pickle(paths.prepared_pkl).copy()
    df = ensure_binary_target(df)

    if WEIGHT not in df.columns:
        df[WEIGHT] = 1.0

    feature_list = [f for f in TIERS[tier_name] if f in df.columns]

    rows = []

    # Sample size row
    no_mde_raw = int((df[TARGET] == 0).sum())
    mde_raw = int((df[TARGET] == 1).sum())
    total_raw = no_mde_raw + mde_raw
    no_mde_pct = round(no_mde_raw / total_raw * 100, 1) if total_raw > 0 else np.nan
    mde_pct = round(mde_raw / total_raw * 100, 1) if total_raw > 0 else np.nan

    rows.append([
        "Total participants",
        "Total respondents in analytic sample",
        f"{no_mde_raw:,} ({no_mde_pct}%)",
        f"{mde_raw:,} ({mde_pct}%)",
        "",
        "",
    ])

    for var in feature_list:
        series = df[var]

        is_categorical = (
            str(series.dtype) == "category"
            or series.dtype == "object"
            or series.nunique(dropna=True) <= 10
        )

        if is_categorical:
            no_mde = weighted_percent(df, var, 0)
            mde = weighted_percent(df, var, 1)

            p_value = approximate_weighted_p_categorical(df, var)

            # Keep the original level types for lookup
            if str(series.dtype) == "category":
                levels = [x for x in series.cat.categories if pd.notna(x)]
            else:
                levels = [x for x in series.dropna().unique()]
                try:
                    levels = sorted(levels)
                except Exception:
                    levels = list(levels)

            for i, level in enumerate(levels):
                no_val = no_mde.get(level, np.nan)
                mde_val = mde.get(level, np.nan)

                level_label = str(level)

                rows.append([
                    pretty_feature_name(f"{var}_{level_label}"),
                    pretty_code_meaning(var, level),
                    f"{no_val:.1f} %" if pd.notna(no_val) else "",
                    f"{mde_val:.1f} %" if pd.notna(mde_val) else "",
                    "Weighted LR test (HC1 robust SE)" if i == 0 else "",
                    format_p_value(p_value) if i == 0 else "",
                ])

        else:
            d0 = df.loc[df[TARGET] == 0, [var, WEIGHT]].dropna()
            d1 = df.loc[df[TARGET] == 1, [var, WEIGHT]].dropna()

            mean0, sd0 = weighted_mean_sd(d0[var], d0[WEIGHT]) if len(d0) else (np.nan, np.nan)
            mean1, sd1 = weighted_mean_sd(d1[var], d1[WEIGHT]) if len(d1) else (np.nan, np.nan)

            p_value = approximate_weighted_p_continuous(df, var)

            rows.append([
                pretty_feature_name(var),
                "",
                f"{mean0:.2f} ± {sd0:.2f}",
                f"{mean1:.2f} ± {sd1:.2f}",
                "Weighted GLM (HC1 robust SE)",
                format_p_value(p_value),
            ])

    table = pd.DataFrame(
        rows,
        columns=[
            "Feature (code)",
            "Code Meaning",
            "No MDE (Weighted %) or Mean ± SD",
            "MDE (Weighted %) or Mean ± SD",
            "Statistical Test",
            "P-value",
        ],
    )

    out_name = f"table1_{tier_name}_Baseline_Characteristics_with_pvalues.csv"
    table.to_csv(paths.results_dir / out_name, index=False)

    note = (
        "P-values are approximate weighted model-based association tests using frequency weights. "
        "They utilize HC1 robust standard errors but are not true complex-survey design-based tests."
    )
    pd.DataFrame({"note": [note]}).to_csv(
        paths.results_dir / f"table1_{tier_name}_Baseline_note.csv",
        index=False,
    )

    return table


# ---------------------------------------------------------------------
# Feature selection intentionally skipped
# ---------------------------------------------------------------------
def run_feature_selection(*args, **kwargs):
    print("[INFO] Feature selection is skipped in this version of the pipeline.")
    print("[INFO] Using pre-defined Tier_1 / Tier_2 / Tier_3 feature lists only.")
    return {}


# ---------------------------------------------------------------------
# Modeling helpers
# ---------------------------------------------------------------------
def ensure_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    out = out[out[TARGET].isin([0, 1])].copy()
    out[TARGET] = out[TARGET].astype(int)
    return out


def get_prepared_data(paths: Paths) -> pd.DataFrame:
    if not paths.prepared_pkl.exists():
        raise FileNotFoundError(f"Prepared data not found: {paths.prepared_pkl}")
    df = pd.read_pickle(paths.prepared_pkl)
    return ensure_binary_target(df)


def split_final_model_data(df: pd.DataFrame):
    y = df[TARGET].astype(int)
    if WEIGHT in df.columns:
        w = pd.to_numeric(df[WEIGHT], errors="coerce").fillna(1.0).astype(float)
    else:
        w = pd.Series(np.ones(len(df)), index=df.index, dtype=float)

    X = df.drop(columns=[TARGET] + ([WEIGHT] if WEIGHT in df.columns else []))
    return train_test_split(X, y, w, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def _select_existing_features(df_columns: Iterable[str], features: List[str], tier_name: str) -> List[str]:
    cols = list(df_columns)
    existing = [f for f in features if f in cols]
    missing = [f for f in features if f not in cols]
    if missing:
        print(f"[WARN] {tier_name}: {len(missing)} features missing from prepared data: {missing}")
    if not existing:
        raise ValueError(f"{tier_name}: none of the requested tier features were found.")
    return existing


def fit_with_optional_sample_weight(model, X, y, sample_weight):
    try:
        model.fit(X, y, clf__sample_weight=sample_weight)
        return model
    except Exception:
        pass
    try:
        model.fit(X, y, sample_weight=sample_weight)
        return model
    except Exception:
        pass
    model.fit(X, y)
    return model


def fit_model_for_family(model, model_name: str, X, y, sample_weight):
    return fit_with_optional_sample_weight(model, X, y, sample_weight)


def predict_proba_with_model(model, X, model_name: str) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def confusion_elements(y_true: np.ndarray, preds: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, preds, sample_weight=sample_weight).ravel()
    return float(tn), float(fp), float(fn), float(tp)


def evaluate_model(y_true: pd.Series, preds: np.ndarray, probs: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_elements(np.asarray(y_true), preds, sample_weight)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity = recall_score(y_true, preds, sample_weight=sample_weight, zero_division=0)
    ppv = precision_score(y_true, preds, sample_weight=sample_weight, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "Accuracy": accuracy_score(y_true, preds, sample_weight=sample_weight),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "AUC": roc_auc_score(y_true, probs, sample_weight=sample_weight),
        "AveragePrecision": average_precision_score(y_true, probs, sample_weight=sample_weight),
        "F1": fbeta_score(y_true, preds, beta=1, sample_weight=sample_weight, zero_division=0),
        "F2": fbeta_score(y_true, preds, beta=2, sample_weight=sample_weight, zero_division=0),
    }


def evaluate_at_threshold(y_true: pd.Series, probs: np.ndarray, threshold: float, sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    base = evaluate_model(y_true, preds, probs, sample_weight)
    base["threshold"] = threshold
    return base

def threshold_sweep(y_true: pd.Series, probs: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> pd.DataFrame:
    thresholds = np.arange(0.10, 0.91, 0.05)
    rows = []
    for t in thresholds:
        rows.append(evaluate_at_threshold(y_true, probs, float(t), sample_weight))
    return pd.DataFrame(rows)


def choose_screening_threshold(metrics_df: pd.DataFrame) -> pd.Series:
    screening_candidates = metrics_df[
        (metrics_df["Sensitivity"] >= 0.90) &
        (metrics_df["NPV"] >= 0.95)
    ]
    if screening_candidates.empty:
        return metrics_df.sort_values(
            ["Sensitivity", "NPV", "Specificity", "F2"],
            ascending=[False, False, False, False],
        ).iloc[0]
    return screening_candidates.sort_values(
        ["Specificity", "F2"],
        ascending=[False, False],
    ).iloc[0]


def choose_locked_threshold_from_training_cv(
    best_model,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
) -> Tuple[float, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=REPORT_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(X_train), dtype=float)

    for tr_idx, val_idx in skf.split(X_train, y_train):
        Xtr = X_train.iloc[tr_idx].copy()
        Xva = X_train.iloc[val_idx].copy()
        ytr = y_train.iloc[tr_idx]
        wtr = w_train.iloc[tr_idx]

        model_clone = clone(best_model)
        Xtr_fit, Xva_fit = Xtr, Xva

        fit_model_for_family(model_clone, model_name, Xtr_fit, ytr, wtr)
        oof_probs[val_idx] = predict_proba_with_model(model_clone, Xva_fit, model_name)

    threshold_df = threshold_sweep(
        y_train.reset_index(drop=True), 
        oof_probs, 
        sample_weight=w_train.reset_index(drop=True)
    )

    best_threshold_row = choose_screening_threshold(threshold_df)
    return float(best_threshold_row["threshold"]), threshold_df


def bootstrap_metric_cis(
    y_true: pd.Series,
    probs: np.ndarray,
    threshold: float,
    sample_weight: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict[str, Tuple[float, float]]:
    metrics_store: Dict[str, List[float]] = {
        "Accuracy": [],
        "Sensitivity": [],
        "Specificity": [],
        "PPV": [],
        "NPV": [],
        "AUC": [],
        "AveragePrecision": [],
        "F1": [],
        "F2": [],
        "Brier": [],
    }

    y = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)
    w = np.asarray(sample_weight) if sample_weight is not None else np.ones_like(y, dtype=float)

    for _ in range(n_boot):
        idx = np.random.randint(0, len(y), len(y))
        y_bs = y[idx]
        p_bs = probs[idx]
        w_bs = w[idx]
        if len(np.unique(y_bs)) < 2:
            continue

        preds_bs = (p_bs >= threshold).astype(int)
        base = evaluate_model(y_bs, preds_bs, p_bs, sample_weight=w_bs)
        base["Brier"] = brier_score_loss(y_bs, p_bs, sample_weight=w_bs)

        for k in metrics_store:
            metrics_store[k].append(float(base[k]))

    out: Dict[str, Tuple[float, float]] = {}
    for k, vals in metrics_store.items():
        if len(vals) == 0:
            out[k] = (np.nan, np.nan)
        else:
            out[k] = tuple(np.percentile(vals, [2.5, 97.5]).tolist())
    return out


def calibration_metrics(y_true: pd.Series, probs: np.ndarray) -> Dict[str, float]:
    out = {"Brier": brier_score_loss(y_true, probs), "Calib_Intercept": np.nan, "Calib_Slope": np.nan}

    if not HAS_STATSMODELS:
        return out

    try:
        y = pd.Series(y_true).astype(float).reset_index(drop=True)
        p = np.clip(pd.Series(probs).astype(float).reset_index(drop=True), 1e-6, 1 - 1e-6)
        lp = np.log(p / (1 - p))

        # intercept: y ~ 1 with offset=lp
        X_int = np.ones((len(y), 1))
        fit_int = sm.GLM(y, X_int, family=sm.families.Binomial(), offset=lp).fit()
        out["Calib_Intercept"] = float(fit_int.params[0])

        # slope: y ~ lp
        X_slope = sm.add_constant(lp)
        fit_slope = sm.GLM(y, X_slope, family=sm.families.Binomial()).fit()
        out["Calib_Slope"] = float(fit_slope.params[1])
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------
def build_logreg_search(X: pd.DataFrame) -> GridSearchCV:
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ],
        remainder="drop",
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "clf__C": [0.1, 0.5, 1.0, 2.0],
        "clf__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(pipe, param_grid=param_grid, scoring="recall", cv=cv, n_jobs=-1, refit=True)


def build_rf_search(X: pd.DataFrame) -> RandomizedSearchCV:
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical_features),
        ],
        remainder="drop",
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    param_dist = {
        "clf__n_estimators": [300, 500],
        "clf__max_depth": [None, 6, 10],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", 0.5],
        "clf__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=15,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
    )


def build_xgb_search(X: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    if not HAS_XGB:
        raise ImportError("xgboost is not installed.")

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical_features),
        ],
        remainder="drop",
    )

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )),
    ])

    param_dist = {
    "clf__n_estimators": [200, 400, 600],
    "clf__max_depth": [3, 4, 5],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__subsample": [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0],
    "clf__min_child_weight": [1, 3],
    "clf__scale_pos_weight": [1.0, scale_pos_weight],
    "clf__reg_alpha": [0.0, 0.01, 0.1],
    "clf__reg_lambda": [1.0, 2.0, 5.0],
    }

    cv = StratifiedKFold(n_splits=SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=15,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
    )


def build_lgbm_search(X: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    if not HAS_LGBM:
        raise ImportError("lightgbm is not installed.")
    
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")), 
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical_features),
        ],
        remainder="drop",
    )
    preprocessor.set_output(transform="pandas")
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
            max_bin=255,
        )),
    ])

    param_dist = {
        "clf__num_leaves": sp_randint(15, 64),
        "clf__max_depth": [-1, 3, 4, 5, 6, 8],
        "clf__learning_rate": sp_uniform(0.01, 0.04),
        "clf__n_estimators": sp_randint(200, 800),
        "clf__min_child_samples": sp_randint(5, 20),
        "clf__subsample": sp_uniform(0.7, 0.3),
        "clf__colsample_bytree": sp_uniform(0.7, 0.3),
        "clf__reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "clf__reg_lambda": [0.0, 0.01, 0.1, 1.0],
        "clf__scale_pos_weight": [1.0, scale_pos_weight],
    }

    cv = StratifiedKFold(n_splits=SEARCH_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=15,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
    )

# ---------------------------------------------------------------------
# Curves for publication figures
# ---------------------------------------------------------------------
def net_benefit(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)
    n = len(y_true)
    out = []

    for pt in thresholds:
        pred_pos = probs >= pt
        tp = np.sum((pred_pos == 1) & (y_true == 1))
        fp = np.sum((pred_pos == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        out.append(nb)

    return np.asarray(out)


def collect_cv_curve_payload(best_model, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, w_train: pd.Series) -> Dict[str, np.ndarray]:
    rkf = RepeatedStratifiedKFold(
        n_splits=REPORT_CV_FOLDS,
        n_repeats=REPORT_CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    fpr_grid = np.linspace(0.0, 1.0, 200)
    recall_grid = np.linspace(0.0, 1.0, 200)
    cal_bins = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    cal_centers = (cal_bins[:-1] + cal_bins[1:]) / 2
    dca_thresholds = np.linspace(0.01, 0.99, 99)

    roc_curves, pr_curves, cal_curves, dca_curves = [], [], [], []
    aucs, aps, prevalences = [], [], []

    for tr_idx, val_idx in rkf.split(X_train, y_train):
        Xtr = X_train.iloc[tr_idx].copy()
        Xva = X_train.iloc[val_idx].copy()
        ytr = y_train.iloc[tr_idx]
        yva = y_train.iloc[val_idx]
        wtr = w_train.iloc[tr_idx]

        model_clone = clone(best_model)
        Xtr_fit, Xva_fit = Xtr, Xva

        fit_model_for_family(model_clone, model_name, Xtr_fit, ytr, wtr)
        probs = predict_proba_with_model(model_clone, Xva_fit, model_name)

        fpr, tpr, _ = roc_curve(yva, probs)
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        roc_curves.append(tpr_interp)
        aucs.append(roc_auc_score(yva, probs))

        precision, recall, _ = precision_recall_curve(yva, probs)
        recall_sorted = recall[::-1]
        precision_sorted = precision[::-1]
        pr_interp = np.interp(recall_grid, recall_sorted, precision_sorted)
        pr_curves.append(pr_interp)
        aps.append(average_precision_score(yva, probs))

        probs_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
        bin_ids = np.digitize(probs_clipped, cal_bins, right=True) - 1
        obs_rates = []
        for b in range(CALIBRATION_BINS):
            mask = bin_ids == b
            if np.any(mask):
                obs_rates.append(np.mean(yva[mask]))
            else:
                obs_rates.append(np.nan)
        cal_curves.append(obs_rates)

        dca_curves.append(net_benefit(yva.to_numpy(), probs, dca_thresholds))
        prevalences.append(float(np.mean(yva)))

    roc_curves = np.asarray(roc_curves)
    pr_curves = np.asarray(pr_curves)
    cal_curves = np.asarray(cal_curves, dtype=float)
    dca_curves = np.asarray(dca_curves)

    return {
        "fpr_grid": fpr_grid,
        "roc_mean": np.nanmean(roc_curves, axis=0),
        "roc_std": np.nanstd(roc_curves, axis=0),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
        "recall_grid": recall_grid,
        "pr_mean": np.nanmean(pr_curves, axis=0),
        "pr_std": np.nanstd(pr_curves, axis=0),
        "ap_mean": float(np.nanmean(aps)),
        "ap_std": float(np.nanstd(aps)),
        "cal_x": cal_centers,
        "cal_mean": np.nanmean(cal_curves, axis=0),
        "cal_std": np.nanstd(cal_curves, axis=0),
        "dca_thresholds": dca_thresholds,
        "dca_mean": np.nanmean(dca_curves, axis=0),
        "dca_std": np.nanstd(dca_curves, axis=0),
        "prevalence_mean": float(np.nanmean(prevalences)),
    }


def save_combined_tier_figures(curve_payloads: Dict[str, Dict[str, Dict[str, np.ndarray]]], figures_dir: Path) -> None:
    for tier_name, model_payloads in curve_payloads.items():
        safe_tier = tier_name.lower()

        # ROC
        plt.figure(figsize=(7.0, 6.0))
        for model_name, payload in model_payloads.items():
            x = payload["fpr_grid"]
            y = payload["roc_mean"]
            s = payload["roc_std"]
            plt.plot(x, y, lw=2, label=f"{model_name} (AUC {payload['auc_mean']:.3f}±{payload['auc_std']:.3f})")
            plt.fill_between(x, np.maximum(0, y - s), np.minimum(1, y + s), alpha=0.15)
        plt.plot([0, 1], [0, 1], linestyle="--", lw=1, color="black")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Comparison — {tier_name}")
        plt.legend(frameon=False, loc="lower right")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{safe_tier}_combined_roc.svg", format="svg", bbox_inches="tight")
        plt.close()

        # PR
        plt.figure(figsize=(7.0, 6.0))
        for model_name, payload in model_payloads.items():
            x = payload["recall_grid"]
            y = payload["pr_mean"]
            s = payload["pr_std"]
            plt.plot(x, y, lw=2, label=f"{model_name} (AP {payload['ap_mean']:.3f}±{payload['ap_std']:.3f})")
            plt.fill_between(x, np.maximum(0, y - s), np.minimum(1, y + s), alpha=0.15)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall Comparison — {tier_name}")
        plt.legend(frameon=False, loc="best")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{safe_tier}_combined_pr.svg", format="svg", bbox_inches="tight")
        plt.close()

        # Calibration
        plt.figure(figsize=(7.0, 6.0))
        for model_name, payload in model_payloads.items():
            x = payload["cal_x"]
            y = payload["cal_mean"]
            s = payload["cal_std"]
            plt.plot(x, y, marker="o", lw=2, label=model_name)
            plt.fill_between(x, np.maximum(0, y - s), np.minimum(1, y + s), alpha=0.15)
        plt.plot([0, 1], [0, 1], linestyle="--", lw=1, color="black")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Observed Event Rate")
        plt.title(f"Calibration Comparison — {tier_name}")
        plt.legend(frameon=False, loc="best")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{safe_tier}_combined_calibration.svg", format="svg", bbox_inches="tight")
        plt.close()

        # Decision curve
        first_payload = next(iter(model_payloads.values()))
        thresholds = first_payload["dca_thresholds"]
        prevalence = first_payload["prevalence_mean"]

        plt.figure(figsize=(7.0, 6.0))
        for model_name, payload in model_payloads.items():
            x = payload["dca_thresholds"]
            y = payload["dca_mean"]
            s = payload["dca_std"]
            plt.plot(x, y, lw=2, label=model_name)
            plt.fill_between(x, y - s, y + s, alpha=0.15)

        treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
        treat_none = np.zeros_like(thresholds)
        plt.plot(thresholds, treat_all, linestyle="--", lw=1, label="Treat All")
        plt.plot(thresholds, treat_none, linestyle=":", lw=1, label="Treat None")

        plt.xlabel("Threshold Probability")
        plt.ylabel("Net Benefit")
        plt.title(f"Decision Curve Analysis — {tier_name}")
        plt.legend(frameon=False, loc="best")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{safe_tier}_combined_decision_curve.svg", format="svg", bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------
# Final model training / evaluation
# ---------------------------------------------------------------------
def run_cv_table(best_model, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, w_train: pd.Series, metrics_order: List[str]) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=REPORT_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        Xtr = X_train.iloc[tr_idx].copy()
        Xval = X_train.iloc[val_idx].copy()
        ytr = y_train.iloc[tr_idx]
        yval = y_train.iloc[val_idx]
        wtr = w_train.iloc[tr_idx]

        model_clone = clone(best_model)
        Xtr_fit, Xval_fit = Xtr, Xval

        fit_model_for_family(model_clone, model_name, Xtr_fit, ytr, wtr)
        preds = model_clone.predict(Xval_fit)
        probs = predict_proba_with_model(model_clone, Xval_fit, model_name)

        fold_metrics = evaluate_model(yval, preds, probs)
        fold_metrics["Fold"] = f"Fold {fold_i}"
        fold_rows.append(fold_metrics)

    fold_df = pd.DataFrame(fold_rows).set_index("Fold")[metrics_order].T
    fold_df["Mean"] = fold_df.mean(axis=1)
    fold_df["Std Dev"] = fold_df.std(axis=1)
    fold_df = fold_df.reset_index().rename(columns={"index": "Metric"})
    return fold_df


def run_model_family(model_name: str, df: pd.DataFrame, paths: Paths) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    X_train, X_test, y_train, y_test, w_train, w_test = split_final_model_data(df)

    tier_results = []
    tier_cv_tables = []
    tier_curve_payloads = {}
    metrics_order = ["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "AUC", "AveragePrecision", "F1", "F2"]

    for tier_name, features in TIERS.items():
        print(f"\n===== {model_name}: {tier_name} =====")
        tier_features = _select_existing_features(X_train.columns, features, tier_name)
        Xtr = X_train[tier_features].copy()
        Xte = X_test[tier_features].copy()

        if model_name == "LogReg":
            search = build_logreg_search(Xtr)
        elif model_name == "RF":
            search = build_rf_search(Xtr)
        elif model_name == "XGB":
            search = build_xgb_search(Xtr, y_train)
        elif model_name == "LGBM":
            search = build_lgbm_search(Xtr, y_train)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        Xtr_fit, Xte_fit = Xtr, Xte

        fit_model_for_family(search, model_name, Xtr_fit, y_train, w_train)
        best_model = search.best_estimator_ if hasattr(search, "best_estimator_") else search
        model_file_prefix = {
            "LogReg": "logreg",
            "RF": "rf",
            "XGB": "xgb",
            "LGBM": "lightgbm",
        }[model_name]

        joblib.dump(best_model, paths.models_dir / f"{model_file_prefix}_{tier_name.lower()}.joblib")

        locked_threshold, train_threshold_df = choose_locked_threshold_from_training_cv(
            best_model=best_model,
            model_name=model_name,
            X_train=Xtr,
            y_train=y_train,
            w_train=w_train,
        )
        train_threshold_df.insert(0, "tier", tier_name)
        train_threshold_df.insert(0, "model", model_name)
        train_threshold_df.to_csv(
            paths.results_dir / f"{model_name.lower()}_{tier_name.lower()}_training_threshold_sweep.csv",
            index=False,
        )

        # Final locked evaluation on test set (Weighted metrics)
        probs_test = predict_proba_with_model(best_model, Xte_fit, model_name)
        preds_default = (probs_test >= 0.50).astype(int)
        preds_locked = (probs_test >= locked_threshold).astype(int)

        metrics_default = evaluate_model(y_test, preds_default, probs_test, sample_weight=w_test)
        metrics_locked = evaluate_model(y_test, preds_locked, probs_test, sample_weight=w_test)

        cal = calibration_metrics(y_test, probs_test)
        metric_cis = bootstrap_metric_cis(y_test, probs_test, threshold=locked_threshold, sample_weight=w_test, n_boot=N_BOOT)

        row_default = {
            "selection": "Default_0.50",
            "tier": tier_name,
            "model": model_name,
            "tuned_for": "recall",
            "n_features": len(tier_features),
            "threshold": 0.50,
            **metrics_default,
            "Brier": cal["Brier"],
            "Calib_Intercept": cal["Calib_Intercept"],
            "Calib_Slope": cal["Calib_Slope"],
            "Best_Params": json.dumps(getattr(search, "best_params_", {}), default=str),
        }

        row_locked = {
            "selection": "Locked_TrainCV_Threshold",
            "tier": tier_name,
            "model": model_name,
            "tuned_for": "recall",
            "n_features": len(tier_features),
            "threshold": locked_threshold,
            **metrics_locked,
            "Brier": cal["Brier"],
            "Calib_Intercept": cal["Calib_Intercept"],
            "Calib_Slope": cal["Calib_Slope"],
            "Accuracy_CI_Low": metric_cis["Accuracy"][0],
            "Accuracy_CI_High": metric_cis["Accuracy"][1],
            "Sensitivity_CI_Low": metric_cis["Sensitivity"][0],
            "Sensitivity_CI_High": metric_cis["Sensitivity"][1],
            "Specificity_CI_Low": metric_cis["Specificity"][0],
            "Specificity_CI_High": metric_cis["Specificity"][1],
            "PPV_CI_Low": metric_cis["PPV"][0],
            "PPV_CI_High": metric_cis["PPV"][1],
            "NPV_CI_Low": metric_cis["NPV"][0],
            "NPV_CI_High": metric_cis["NPV"][1],
            "AUC_CI_Low": metric_cis["AUC"][0],
            "AUC_CI_High": metric_cis["AUC"][1],
            "AveragePrecision_CI_Low": metric_cis["AveragePrecision"][0],
            "AveragePrecision_CI_High": metric_cis["AveragePrecision"][1],
            "F1_CI_Low": metric_cis["F1"][0],
            "F1_CI_High": metric_cis["F1"][1],
            "F2_CI_Low": metric_cis["F2"][0],
            "F2_CI_High": metric_cis["F2"][1],
            "Brier_CI_Low": metric_cis["Brier"][0],
            "Brier_CI_High": metric_cis["Brier"][1],
            "Best_Params": json.dumps(getattr(search, "best_params_", {}), default=str),
        }

        tier_results.extend([row_default, row_locked])

        cv_table = run_cv_table(best_model, model_name, Xtr, y_train, w_train, metrics_order)
        cv_table.insert(0, "Tier", tier_name)
        cv_table.insert(0, "Model", model_name)
        tier_cv_tables.append(cv_table)

        tier_curve_payloads[tier_name] = collect_cv_curve_payload(best_model, model_name, Xtr, y_train, w_train)

    results_df = pd.DataFrame(tier_results)
    cv_results_df = pd.concat(tier_cv_tables, ignore_index=True)

    model_to_prefix = {
        "LogReg": "logreg",
        "RF": "rf",
        "XGB": "xgb",
        "LGBM": "lightgbm",
    }
    prefix = model_to_prefix[model_name]
    results_df.to_csv(paths.results_dir / f"{prefix}_two_rows_per_tier.csv", index=False)
    cv_results_df.to_csv(paths.results_dir / f"{prefix}_{SEARCH_CV_FOLDS}fold_threshold0.50_by_tier.csv", index=False)

    primary_subset = results_df[results_df["selection"] == "Locked_TrainCV_Threshold"].copy()
    primary_subset.to_csv(paths.results_dir / f"{prefix}_primary_locked_results.csv", index=False)

    return results_df, cv_results_df, tier_curve_payloads


# ---------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------
def create_model_comparison_tables(paths: Paths, included_models: List[str]) -> None:
    prefix_map = {
        "LogReg": "logreg",
        "RF": "rf",
        "XGB": "xgb",
        "LGBM": "lightgbm",
    }

    metrics_keep = ["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "AUC", "AveragePrecision", "F1", "F2"]

    all_two_row_dfs = []
    all_cv_dfs = []

    for model in included_models:
        prefix = prefix_map[model]
        two_row_path = paths.results_dir / f"{prefix}_two_rows_per_tier.csv"
        cv_path = paths.results_dir / f"{prefix}_{SEARCH_CV_FOLDS}fold_threshold0.50_by_tier.csv"

        if two_row_path.exists():
            all_two_row_dfs.append(pd.read_csv(two_row_path))
        if cv_path.exists():
            all_cv_dfs.append(pd.read_csv(cv_path))

    if all_two_row_dfs:
        all_models = pd.concat(all_two_row_dfs, ignore_index=True)
        all_models.to_csv(paths.results_dir / "all_models_by_tier_thresholds.csv", index=False)

        means = (
            all_models[all_models["selection"] == "Default_0.50"]
            .sort_values(["model", "tier"])
            .loc[:, ["model", "tier", "n_features", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "AUC", "AveragePrecision", "F1", "F2", "Brier", "Calib_Intercept", "Calib_Slope"]]
            .reset_index(drop=True)
        )
        means.to_csv(paths.results_dir / "model_comparison_table_means.csv", index=False)

        primary_results = (
            all_models[all_models["selection"] == "Locked_TrainCV_Threshold"]
            .sort_values(["tier", "model"])
            .loc[:, [
                "tier", "model", "n_features", "threshold",
                "Accuracy", "Accuracy_CI_Low", "Accuracy_CI_High",
                "Sensitivity", "Sensitivity_CI_Low", "Sensitivity_CI_High",
                "Specificity", "Specificity_CI_Low", "Specificity_CI_High",
                "PPV", "PPV_CI_Low", "PPV_CI_High",
                "NPV", "NPV_CI_Low", "NPV_CI_High",
                "AUC", "AUC_CI_Low", "AUC_CI_High",
                "AveragePrecision", "AveragePrecision_CI_Low", "AveragePrecision_CI_High",
                "F1", "F1_CI_Low", "F1_CI_High",
                "F2", "F2_CI_Low", "F2_CI_High",
                "Brier", "Brier_CI_Low", "Brier_CI_High",
                "Calib_Intercept", "Calib_Slope",
            ]]
            .reset_index(drop=True)
        )
        primary_results.to_csv(paths.results_dir / "primary_results_locked_thresholds.csv", index=False)

    if all_cv_dfs:
        cv_all = pd.concat(all_cv_dfs, ignore_index=True)
        cv_all = cv_all[cv_all["Metric"].isin(metrics_keep)].copy()
        cv_all.to_csv(paths.results_dir / "all_models_comparison.csv", index=False)


# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------
def run_shap_for_tree_models(df: pd.DataFrame, paths: Paths, models_to_use: Optional[List[str]] = None) -> None:
    if not HAS_SHAP:
        print("[WARN] shap is not installed; skipping SHAP analysis.")
        return

    if models_to_use is None:
        models_to_use = ["LGBM", "XGB"]

    X_train, X_test, y_train, y_test, w_train, w_test = split_final_model_data(df)
    all_importances = []

    for model_name in models_to_use:
        for tier_name, features in TIERS.items():
            tier_features = _select_existing_features(X_test.columns, features, tier_name)
            model_file_prefix = {
                "RF": "rf",
                "XGB": "xgb",
                "LGBM": "lightgbm",
            }[model_name]

            model_path = paths.models_dir / f"{model_file_prefix}_{tier_name.lower()}.joblib"
            if not model_path.exists():
                print(f"[WARN] Missing model for SHAP: {model_path}")
                continue

            model = joblib.load(model_path)
            if hasattr(model, "best_estimator_"):
                model = model.best_estimator_

            Xte = X_test[tier_features].copy()
            clf = model
            transformed = Xte
            feature_names = tier_features

            if hasattr(model, "named_steps"):
                pre = model.named_steps.get("preprocessor")
                clf = model.named_steps.get("clf")
                if pre is not None:
                    transformed = pre.transform(Xte)
                    try:
                        feature_names = pre.get_feature_names_out().tolist()
                    except Exception:
                        feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]

            try:
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(transformed)
                if isinstance(shap_values, list):
                    shap_values = shap_values[-1]
            except Exception as e:
                print(f"[WARN] SHAP failed for {model_name} / {tier_name}: {e}")
                continue

            plt.figure(figsize=(9, 6))
            shap.summary_plot(shap_values, transformed, feature_names=feature_names, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance — {model_name} — {tier_name}")
            plt.tight_layout()
            plt.savefig(paths.figures_dir / f"{model_name.lower()}_shap_importance_{tier_name}.svg", format="svg", bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(9, 6))
            shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary — {model_name} — {tier_name}")
            plt.tight_layout()
            plt.savefig(paths.figures_dir / f"{model_name.lower()}_shap_beeswarm_{tier_name}.svg", format="svg", bbox_inches="tight")
            plt.close()

            mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)
            shap_importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
            shap_importance["model"] = model_name
            shap_importance["tier"] = tier_name
            shap_importance["rank"] = np.arange(1, len(shap_importance) + 1)
            all_importances.append(shap_importance)

    if all_importances:
        out = pd.concat(all_importances, ignore_index=True)
        out = out[["model", "tier", "rank", "feature", "mean_abs_shap"]]
        out.to_csv(paths.results_dir / "tree_model_shap_importance.csv", index=False)


def run_shap_dependence_panels_for_tier3(
    df: pd.DataFrame,
    paths: Paths,
    models_to_use: Optional[List[str]] = None,
    max_display: int = 8,
    sample_n: int = 5000,
) -> None:
    if not HAS_SHAP:
        print("[WARN] shap is not installed; skipping SHAP dependence panels.")
        return

    if models_to_use is None:
        models_to_use = ["RF", "XGB", "LGBM"]

    X_train, X_test, y_train, y_test, w_train, w_test = split_final_model_data(df)

    tier_name = "Tier_3_Personalized"
    tier_features = _select_existing_features(X_test.columns, TIERS[tier_name], tier_name)

    for model_name in models_to_use:
        model_file_prefix = {
            "RF": "rf",
            "XGB": "xgb",
            "LGBM": "lightgbm",
        }[model_name]

        model_path = paths.models_dir / f"{model_file_prefix}_{tier_name.lower()}.joblib"
        if not model_path.exists():
            print(f"[WARN] Missing model for SHAP dependence: {model_path}")
            continue

        print(f"[SHAP] Dependence plots for {model_name} / {tier_name}")

        model = joblib.load(model_path)
        if hasattr(model, "best_estimator_"):
            model = model.best_estimator_

        Xte = X_test[tier_features].copy()

        if len(Xte) > sample_n:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(Xte), size=sample_n, replace=False)
            Xte = Xte.iloc[idx].copy()

        clf = model
        transformed = Xte
        feature_names = tier_features

        if hasattr(model, "named_steps"):
            pre = model.named_steps.get("preprocessor")
            clf = model.named_steps.get("clf")
            if pre is not None:
                transformed = pre.transform(Xte)
                try:
                    feature_names = pre.get_feature_names_out().tolist()
                except Exception:
                    feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]

        if hasattr(transformed, "toarray"):
            transformed_for_shap = transformed.toarray()
        else:
            transformed_for_shap = np.asarray(transformed)

        try:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(transformed_for_shap)

            if isinstance(shap_values, list):
                shap_values = shap_values[-1]

            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, -1]
            elif shap_values.ndim != 2:
                raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")
            
        except Exception as e:
            print(f"[WARN] SHAP dependence failed for {model_name}: {e}")
            continue

        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:max_display].astype(int).tolist()
        top_features = [feature_names[i] for i in top_idx]

        plot_X = pd.DataFrame(transformed_for_shap, columns=feature_names)

        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()

        for ax_i, feat_name in enumerate(top_features):
            plt.sca(axes[ax_i])
            try:
                shap.dependence_plot(
                    feat_name,
                    shap_values,
                    plot_X,
                    interaction_index="auto",
                    ax=axes[ax_i],
                    show=False,
                )
                axes[ax_i].set_title(pretty_feature_name(feat_name), fontsize=10)
            except Exception as e:
                axes[ax_i].text(
                    0.5, 0.5,
                    f"Plot failed:\n{feat_name}",
                    ha="center", va="center", fontsize=10
                )
                axes[ax_i].set_axis_off()

        for j in range(len(top_features), len(axes)):
            axes[j].set_axis_off()

        fig.suptitle(f"SHAP Dependence Plots — Top {len(top_features)} Features — {model_name} — {tier_name}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(
            paths.figures_dir / f"{model_name.lower()}_tier3_shap_dependence_top{max_display}.svg",
            format="svg",
            bbox_inches="tight",
        )
        plt.close()

        top_table = pd.DataFrame({
            "feature": feature_names,
            "feature_label": [pretty_feature_name(f) for f in feature_names],
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        top_table.to_csv(
            paths.results_dir / f"{model_name.lower()}_tier3_shap_top_features.csv",
            index=False,
        )

# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def run_pipeline(
    paths: Paths,
    run_cleaning: bool,
    run_preparation: bool,
    run_baseline: bool,
    run_models: bool,
    run_shap_flag: bool,
    run_pdp: bool = False,
    pdp_model_name: str = "RF",
    pdp_tier_name: str = "Tier_3_Personalized",
    pdp_features: Optional[List[str]] = None,
) -> None:
    configure_publication_plots()
    export_tier_csvs(paths)

    if run_cleaning:
        print("\n[1/4] Cleaning raw data...")
        clean_nsduh(paths.raw_data, paths.cleaned_pkl)
        print(f"Saved cleaned data -> {paths.cleaned_pkl}")

    if run_preparation:
        print("\n[2/4] Preparing features...")
        if not paths.cleaned_pkl.exists():
            raise FileNotFoundError("Run cleaning first or provide cleaned pickle.")
        cleaned_df = pd.read_pickle(paths.cleaned_pkl)
        prepare_features(cleaned_df, paths.prepared_pkl)
        print(f"Saved prepared data -> {paths.prepared_pkl}")

    if run_baseline:
        print("\n[3/4] Building baseline characteristics table with approximate weighted p-values...")
        build_baseline_characteristics_table(paths, tier_name="Tier_3_Personalized")
        print(f"Saved -> {paths.results_dir / 'table1_Tier_3_Personalized_Baseline_Characteristics_with_pvalues.csv'}")

    if run_models:
        print("\n[4/4] Training final models by tier using existing tier definitions...")
        prepared_df = get_prepared_data(paths)

        model_order = ["LogReg", "RF"]
        if HAS_XGB:
            model_order.append("XGB")
        else:
            print("[WARN] xgboost not installed; skipping XGB.")
        if HAS_LGBM and HAS_SCIPY:
            model_order.append("LGBM")
        else:
            print("[WARN] lightgbm and/or scipy not installed; skipping LGBM.")

        curve_payloads_all = {tier_name: {} for tier_name in TIERS.keys()}

        for model_name in model_order:
            _, _, model_curve_payloads = run_model_family(model_name, prepared_df, paths)
            for tier_name, payload in model_curve_payloads.items():
                curve_payloads_all[tier_name][model_name] = payload

        create_model_comparison_tables(paths, model_order)
        save_combined_tier_figures(curve_payloads_all, paths.figures_dir)

        print(f"Saved model comparison tables -> {paths.results_dir}")
        print(f"Saved combined publication figures -> {paths.figures_dir}")

        if run_shap_flag:
            print("\n[SHAP] Running SHAP for tree models...")
            run_shap_for_tree_models(prepared_df, paths)
            
            print("\n[SHAP] Running Tier 3 SHAP dependence panels...")
            run_shap_dependence_panels_for_tier3(
                prepared_df,
                paths,
                models_to_use=["RF", "XGB", "LGBM"],
                max_display=8,
                sample_n=5000,
            )
        
        if run_pdp:
            pdp_features_to_use = pdp_features or TIERS[pdp_tier_name]
            pdp_models_to_run: List[str] = []
            for candidate in ["RF", "XGB", "LGBM"]:
                model_path = paths.models_dir / ({"RF": "rf", "XGB": "xgb", "LGBM": "lightgbm"}[candidate] + f"_{pdp_tier_name.lower()}.joblib")
                if model_path.exists():
                    pdp_models_to_run.append(candidate)
            if not pdp_models_to_run and pdp_model_name:
                pdp_models_to_run = [pdp_model_name]

            for pdp_model in pdp_models_to_run:
                run_partial_dependence_plots(
                    df=prepared_df,
                    tier_name=pdp_tier_name,
                    paths=paths,
                    features_to_plot=pdp_features_to_use,
                    model_name=pdp_model,
                )

    print("\nPipeline complete.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidated NSDUH MDE publication pipeline (feature selection skipped, no SVM, SVG figures only)."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory where data/, results/, models/, tiers/ live.",
    )
    parser.add_argument(
        "--raw-data",
        type=str,
        default="data/raw/NSDUH_2023_Tab.txt",
        help="Path to raw NSDUH tab-delimited file.",
    )

    parser.add_argument("--run-all", action="store_true", help="Run cleaning, preparation, baseline table, final modeling, and optional SHAP.")
    parser.add_argument("--run-cleaning", action="store_true")
    parser.add_argument("--run-preparation", action="store_true")
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--run-models", action="store_true")
    parser.add_argument("--run-shap", action="store_true")
    parser.add_argument("--run-pdp", action="store_true", help="Generate partial dependence plots after model fitting.")
    parser.add_argument("--pdp-model", type=str, default="RF", choices=["RF", "XGB", "LGBM"], help="Fallback model to use for PDP generation if auto-detection finds no saved model files.")
    parser.add_argument("--pdp-tier", type=str, default="Tier_3_Personalized", choices=list(TIERS.keys()), help="Tier to use for PDP generation.")
    parser.add_argument("--pdp-features", nargs="*", default=None, help="Optional list of feature names to plot for PDP. Defaults to a curated Tier 3 set.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    raw_data = Path(args.raw_data).resolve() if Path(args.raw_data).is_absolute() else (project_root / args.raw_data).resolve()
    paths = make_paths(project_root, raw_data)

    if args.run_all:
        run_cleaning = True
        run_preparation = True
        run_baseline = True
        run_models = True
        run_shap_flag = args.run_shap
        run_pdp = args.run_pdp
    else:
        run_cleaning = args.run_cleaning
        run_preparation = args.run_preparation
        run_baseline = args.run_baseline
        run_models = args.run_models
        run_shap_flag = args.run_shap
        run_pdp = args.run_pdp

    if not any([run_cleaning, run_preparation, run_baseline, run_models]):
        print("No stage selected. Use --run-all or one or more stage flags.")
        sys.exit(1)

    if run_cleaning or run_baseline:
        if not raw_data.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_data}")

    run_pipeline(
        paths=paths,
        run_cleaning=run_cleaning,
        run_preparation=run_preparation,
        run_baseline=run_baseline,
        run_models=run_models,
        run_shap_flag=run_shap_flag,
        run_pdp=run_pdp,
        pdp_model_name=args.pdp_model,
        pdp_tier_name=args.pdp_tier,
        pdp_features=args.pdp_features,
    )


if __name__ == "__main__":
    main()