#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference for the trained clinico-ResNet penalized Cox (Coxnet) model.

Given RAW clinical + ResNet feature spreadsheets (same format as the training
inputs), this script reproduces the full training preprocessing pipeline using
the *fitted* transformers saved during training, runs the trained Coxnet model,
and outputs a Risk Score and Risk Group (High / Low) per subject.

Pipeline replayed (transform-only, no refitting):
  1. Data wrangling   -> rename clinical columns, normalize ResNet SubjectIDs
  2. Feature engineer -> label/NF1 encode, CountEncoder, StandardScaler(s),
                         extent normalization, imputer, variance + correlation
                         feature removal (all using saved artifacts)
  3. Predict          -> estimator.predict() == linear predictor == Risk Score
  4. Stratify         -> Risk Group = High if score > discovery median else Low

Usage (from any working directory):
    python inference.py

  Or override any path via CLI flags:
    python inference.py --input-dir /path/to/spreadsheets --output-dir /path/to/results

  Input spreadsheets required (see LGG_inference/ for the expected column format):
    Clinical_Features.xlsx  -- subject clinical variables
    ResNet_Features.xlsx    -- ResNet features extracted from T2w MRI

  The preprocessing artifacts (encoder.pkl, scalers, imputer, remover, dropper)
  and trained model (estimator.pkl, risk_threshold.pkl) ship with this repo and
  are located next to this script by default — no edits needed for standard use.
"""

import os
import re
import pickle
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Directory this script lives in — all default paths resolve relative to it,
# so the script works regardless of where you launch Python from.
SCRIPT_DIR = Path(__file__).resolve().parent

# ============================================================
# CONFIG
# ------------------------------------------------------------
# All paths default to locations relative to this script's directory.
# Override any entry by passing the corresponding CLI flag (see --help),
# or by editing the values below.
# ============================================================
CONFIG = {
    # ---- directories ----
    # Default: all artifacts and model files live next to this script.
    # Pass CLI flags to override (python inference.py --help).
    "input_dir":     "LGG_inference",          # folder with the two input .xlsx files
    "artifact_dir":  "",                        # preprocessing .pkl files (same dir as script)
    "model_dir":     "outputs/models",          # estimator.pkl + risk_threshold.pkl
    "threshold_dir": "outputs/models",          # risk_threshold.pkl (same as model_dir)
    "output_dir":    "LGG_inference",           # where inference_risk_results.csv is written

    # ---- file names (no need to change) ----
    "clinical_xlsx":       "Clinical_Features.xlsx",
    "resnet_xlsx":         "ResNet_Features.xlsx",
    "encoder_pkl":         "encoder.pkl",           # CountEncoder (Tumor Location)
    "scaler_clinical_pkl": "scaler_clinical.pkl",   # StandardScaler (Age, Tumor Location)
    "imputer_pkl":         "imputer.pkl",            # SimpleImputer (ResNet features)
    "scaler_radiomic_pkl": "scaler_radiomic.pkl",   # StandardScaler (ResNet features)
    "remover_pkl":         "remover.pkl",            # VarianceThreshold (ResNet features)
    "dropper_pkl":         "dropper.pkl",            # correlated-feature list
    "estimator_pkl":       "estimator.pkl",
    "risk_threshold_pkl":  "risk_threshold.pkl",
    "output_csv":          "inference_risk_results.csv",
}



def _resolve(path):
    """Make a path absolute, resolved relative to this script's directory."""
    p = Path(path)
    return str(p) if p.is_absolute() else str(SCRIPT_DIR / p)


def build_paths(cfg):
    """Expand the directory + filename entries into full, resolved file paths."""
    art = cfg["artifact_dir"]
    mdl = cfg["model_dir"]
    thr = cfg.get("threshold_dir", cfg["model_dir"])
    inp = cfg["input_dir"]
    out = cfg["output_dir"]
    return {
        "clinical_xlsx":       _resolve(os.path.join(inp, cfg["clinical_xlsx"])),
        "resnet_xlsx":         _resolve(os.path.join(inp, cfg["resnet_xlsx"])),
        "encoder_pkl":         _resolve(os.path.join(art, cfg["encoder_pkl"])),
        "scaler_clinical_pkl": _resolve(os.path.join(art, cfg["scaler_clinical_pkl"])),
        "imputer_pkl":         _resolve(os.path.join(art, cfg["imputer_pkl"])),
        "scaler_radiomic_pkl": _resolve(os.path.join(art, cfg["scaler_radiomic_pkl"])),
        "remover_pkl":         _resolve(os.path.join(art, cfg["remover_pkl"])),
        "dropper_pkl":         _resolve(os.path.join(art, cfg["dropper_pkl"])),
        "estimator_pkl":       _resolve(os.path.join(mdl, cfg["estimator_pkl"])),
        "risk_threshold_pkl":  _resolve(os.path.join(thr, cfg["risk_threshold_pkl"])),
        "output_csv":          _resolve(os.path.join(out, cfg["output_csv"])),
    }


def preflight(paths):
    """Report ALL missing required inputs at once before doing any work."""
    required = [k for k in paths if k != "output_csv"]
    missing = [(k, paths[k]) for k in required if not os.path.exists(paths[k])]
    if missing:
        lines = "\n".join(f"  - {k:20s} -> {p}" for k, p in missing)
        raise FileNotFoundError(
            "The following required files were not found:\n" + lines +
            "\n\nFix the *_DIR entries in CONFIG (or pass --input-dir / "
            "--artifact-dir / --model-dir). The preprocessing .pkl files are "
            "written by the Feature Engineering notebook; estimator.pkl and "
            "risk_threshold.pkl by the Predictive Modeling notebook."
        )

# Deterministic encodings copied verbatim from the Feature Engineering notebook.
LABEL_MAPS = {
    "Sex": {"Female": 0, "Male": 1},
    "Extent of Tumor Resection": {
        "Not Applicable": 0,
        "Unavailable": 0,
        "Biopsy only": 1,
        "Partial resection": 2,
        "Gross/Near total resection": 3,
    },
    "Chemotherapy": {
        "Yes": 1, "No": 0, "Not Applicable": 0,
        "Not Reported": 0, "Unavailable": 0,
    },
    "Radiation": {
        "Yes": 1, "No": 0, "Not Applicable": 0,
        "Not Reported": 0, "Unavailable": 0,
    },
}
EXTENT_NORMALIZER = 3                       # Extent of Tumor Resection / 3
NF1_POSITIVE = "Neurofibromatosis, Type 1 (NF-1)"
CLINICAL_MODEL_FEATURES = [                 # order produced by training pipeline
    "Sex", "Age at Diagnosis", "Tumor Location", "NF1",
    "Extent of Tumor Resection", "Chemotherapy", "Radiation",
]


# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------
def _load(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {what}: '{path}'. Update CONFIG to point at the artifact "
            f"saved during training."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------
# Stage 1: Data wrangling (mirrors notebook 1)
# ------------------------------------------------------------
def wrangle_clinical(path):
    """Read raw clinical spreadsheet, select + rename to model column names."""
    df = pd.read_excel(path)

    rename = {
        "legal_sex": "Sex",
        "age_at_event_days": "Age at Diagnosis",
        "consolidated_tumor_locations": "Tumor Location",
        "cancer_predisposition": "NF1",
        "extent_of_tumor_resection": "Extent of Tumor Resection",
        "chemotherapy": "Chemotherapy",
        "radiation": "Radiation",
    }
    keep_raw = ["SubjectID"] + list(rename.keys())
    missing = [c for c in keep_raw if c not in df.columns]
    if missing:
        raise KeyError(f"Clinical spreadsheet is missing expected columns: {missing}")

    df = df[keep_raw].rename(columns=rename)
    df = df.set_index("SubjectID")
    return df


def wrangle_resnet(path, valid_subjects):
    """Read raw ResNet spreadsheet, normalize SubjectIDs, filter, index."""
    df = pd.read_excel(path)

    df["SubjectID"] = df["SubjectID"].apply(
        lambda x: (m := re.match(r"^(C\d+|sub\d+)", str(x))) and m.group()
    )
    df = df[df["SubjectID"].isin(valid_subjects)]
    df = df.set_index("SubjectID")
    if "Session" in df.columns:
        df = df.drop(columns=["Session"])
    return df


# ------------------------------------------------------------
# Stage 2: Feature engineering (mirrors notebook 2, transform-only)
# ------------------------------------------------------------
def engineer_clinical(df, encoder, scaler_clinical):
    """Apply the exact clinical feature engineering using saved transformers."""
    X = df.copy()

    # label encodings (deterministic maps)
    X = X.replace(to_replace=LABEL_MAPS)

    # NF1 -> binary
    X["NF1"] = X["NF1"].apply(lambda v: 1 if v == NF1_POSITIVE else 0)

    # Tumor Location: count encoding (fitted on discovery) -> then scaling
    X[["Tumor Location"]] = encoder.transform(X[["Tumor Location"]])
    if X["Tumor Location"].isna().any():
        warnings.warn(
            "CountEncoder produced NaN for a Tumor Location value unseen during "
            "training. Check the location strings in your inference data."
        )

    # standardize Age + Tumor Location (same column order as training)
    sc_cols = ["Age at Diagnosis", "Tumor Location"]
    X[sc_cols] = scaler_clinical.transform(X[sc_cols])

    # normalize Extent of Tumor Resection
    X["Extent of Tumor Resection"] = (
        pd.to_numeric(X["Extent of Tumor Resection"], errors="coerce")
        / EXTENT_NORMALIZER
    )

    # enforce the training column order and ensure everything is numeric
    X = X[CLINICAL_MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")

    # surface any blank / unmapped categorical values that became NaN
    nan_mask = X.isna()
    if nan_mask.any().any():
        bad = [
            f"{idx} -> {col}"
            for idx in X.index
            for col in X.columns
            if nan_mask.loc[idx, col]
        ]
        warnings.warn(
            "Clinical feature(s) are blank or contain a value not seen during "
            "training, which will yield a NaN risk score for the affected "
            "subject(s). Review these: " + "; ".join(bad)
        )
    return X


def engineer_resnet(df, imputer, scaler_radiomic, remover, dropper):
    """Apply the exact ResNet feature engineering using saved transformers."""
    X = df.copy()

    # impute -> standardize (preserve column order)
    X.loc[:, :] = imputer.transform(X)
    X.loc[:, :] = scaler_radiomic.transform(X)

    # variance threshold: keep only columns selected during training
    keep_idx = list(remover.get_support(indices=True))
    X = X[X.columns[keep_idx]]

    # drop correlated features identified during training
    drop_cols = [c for c in dropper if c in X.columns]
    X = X.drop(columns=drop_cols)
    return X


# ------------------------------------------------------------
# Stage 3 + 4: Predict + stratify
# ------------------------------------------------------------
def predict_risk(X_all, estimator, median_threshold, drop_incomplete=False):
    # Reorder to the columns the estimator was trained on, when available.
    feat_names = getattr(estimator, "feature_names_in_", None)
    if feat_names is not None:
        feat_names = list(feat_names)
        missing = [c for c in feat_names if c not in X_all.columns]
        if missing:
            raise KeyError(
                f"Engineered features are missing columns the model expects: "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
            )
        X_all = X_all[feat_names]

    # ---- guard against NaNs (Coxnet rejects them) ----
    nan_mask = X_all.isna()
    if nan_mask.any().any():
        bad_rows = nan_mask.any(axis=1)
        report = [
            f"  {idx}: " + ", ".join(X_all.columns[nan_mask.loc[idx].values])
            for idx in X_all.index[bad_rows]
        ]
        detail = "\n".join(report)
        if drop_incomplete:
            warnings.warn(
                f"Dropping {int(bad_rows.sum())} subject(s) with missing feature "
                f"value(s) before scoring:\n{detail}"
            )
            X_all = X_all.loc[~bad_rows]
            if X_all.empty:
                raise RuntimeError("All subjects had missing features; nothing to score.")
        else:
            raise ValueError(
                "Cannot score: the following subject(s) have missing feature "
                "value(s) (the trained pipeline does not impute clinical "
                "features):\n" + detail +
                "\n\nFix options:\n"
                "  1. Fill the missing value(s) in the source spreadsheet, or\n"
                "  2. re-run with --drop-incomplete to skip these subjects."
            )

    scores = estimator.predict(X_all)

    out = pd.DataFrame({"Risk Score": np.asarray(scores, dtype=float)},
                       index=X_all.index)
    out["Risk Group"] = np.where(
        out["Risk Score"] > median_threshold, "High", "Low"
    )
    return out


# ------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------
def run(cfg):
    paths = build_paths(cfg)
    preflight(paths)            # report all missing files up front

    # ---- load fitted artifacts ----
    encoder         = _load(paths["encoder_pkl"],         "clinical CountEncoder")
    scaler_clinical = _load(paths["scaler_clinical_pkl"], "clinical StandardScaler")
    imputer         = _load(paths["imputer_pkl"],         "ResNet SimpleImputer")
    scaler_radiomic = _load(paths["scaler_radiomic_pkl"], "ResNet StandardScaler")
    remover         = _load(paths["remover_pkl"],         "ResNet VarianceThreshold")
    dropper         = _load(paths["dropper_pkl"],         "ResNet correlated-feature list")
    estimator       = _load(paths["estimator_pkl"],       "trained Coxnet estimator")

    thr_obj = _load(paths["risk_threshold_pkl"], "risk threshold")
    median_threshold = (
        thr_obj["median_threshold"] if isinstance(thr_obj, dict) else float(thr_obj)
    )

    # ---- stage 1: wrangle ----
    clin_raw   = wrangle_clinical(paths["clinical_xlsx"])
    resnet_raw = wrangle_resnet(paths["resnet_xlsx"], valid_subjects=set(clin_raw.index))

    # ---- stage 2: engineer ----
    X_clin   = engineer_clinical(clin_raw, encoder, scaler_clinical)
    X_resnet = engineer_resnet(resnet_raw, imputer, scaler_radiomic, remover, dropper)

    # merge on SubjectID (clinical columns first, then ResNet -- training order)
    X_all = X_clin.merge(X_resnet, left_index=True, right_index=True, how="inner")
    if X_all.empty:
        raise RuntimeError(
            "No subjects remained after merging clinical and ResNet features. "
            "Check that SubjectIDs match between the two spreadsheets."
        )

    # ---- stage 3 + 4: predict + stratify ----
    results = predict_risk(X_all, estimator, median_threshold,
                           drop_incomplete=cfg.get("drop_incomplete", False))
    os.makedirs(os.path.dirname(paths["output_csv"]), exist_ok=True)
    results.to_csv(paths["output_csv"], index=True)

    print(f"Median threshold (from discovery): {median_threshold:.6f}")
    print(f"Subjects scored: {len(results)}")
    print(results)
    print(f"\nSaved -> {paths['output_csv']}")
    return results


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run DL-M1 (clinico-ResNet Coxnet) risk inference on new subjects.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: use paths shipped with the repo
  python inference.py

  # Custom input spreadsheets in a different folder
  python inference.py --input-dir /data/my_cohort --output-dir /results

  # Skip subjects that have missing feature values
  python inference.py --drop-incomplete
""",
    )
    p.add_argument("--input-dir",    dest="input_dir",
                   help="Folder containing Clinical_Features.xlsx and ResNet_Features.xlsx")
    p.add_argument("--artifact-dir", dest="artifact_dir",
                   help="Folder with preprocessing .pkl files (default: script directory)")
    p.add_argument("--model-dir",    dest="model_dir",
                   help="Folder with estimator.pkl and risk_threshold.pkl (default: outputs/models)")
    p.add_argument("--output-dir",   dest="output_dir",
                   help="Folder where inference_risk_results.csv will be written")
    p.add_argument("--clinical",     dest="clinical_xlsx",
                   help="Filename of the clinical spreadsheet (default: Clinical_Features.xlsx)")
    p.add_argument("--resnet",       dest="resnet_xlsx",
                   help="Filename of the ResNet features spreadsheet (default: ResNet_Features.xlsx)")
    p.add_argument("--out",          dest="output_csv",
                   help="Output CSV filename (default: inference_risk_results.csv)")
    p.add_argument("--drop-incomplete", dest="drop_incomplete",
                   action="store_true", default=None,
                   help="Skip subjects with any missing feature instead of raising an error.")
    return p.parse_args()


if __name__ == "__main__":
    cfg = dict(CONFIG)
    args = _parse_args()
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    run(cfg)