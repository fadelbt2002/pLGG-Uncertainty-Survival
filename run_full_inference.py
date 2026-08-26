#!/usr/bin/env python3
"""
run_full_inference.py — Full pLGG Risk Inference Pipeline with 95% Bootstrap CIs
==================================================================================
End-to-end inference for new subjects: T2w MRI → clinical risk report.

Steps
-----
  1  ResNet feature extraction
       T2w NIfTI folder → layer3 GAP+GMP features (512-dim)

  2  DL-M1 risk score + 95% CI
       Clinical + ResNet features → penalized Cox model
       Risk Score, Risk Group (High/Low), CI from 1000 bootstrap resamples

  3  DL-M2 risk score + 95% CI  [requires --mol_subtype]
       DL-M1 score + molecular subtype → late-fusion CoxPH
       Fusion Risk Score, Fusion Risk Group, CI from 1000 bootstrap entries

  4a DL-M1 treatment scenarios + 95% CI
       4 resection × chemo combinations → survival curves with CI bands

  4b DL-M2 treatment scenarios + 95% CI  [requires --mol_subtype]
       Same 4 scenarios propagated through the fusion model

Clinical_Features.xlsx format (9 columns)
------------------------------------------
  SubjectID                   — must match NIfTI filename prefix
  legal_sex                   — Female | Male
  age_at_event_days           — integer (days at imaging)
  consolidated_tumor_locations— Cerebellar | Lobar | Suprasellar | Brain Stem |
                                Multi regional | Intra Ventricular | Thalamus |
                                Basal Ganglia | Other
  cancer_predisposition       — None documented |
                                Neurofibromatosis, Type 1 (NF-1)
  extent_of_tumor_resection   — Biopsy only | Partial resection |
                                Gross/Near total resection | Not Applicable
  chemotherapy                — Yes | No
  radiation                   — Yes | No
  molecular_subtype           — per-subject molecular subtype (see choices below);
                                leave blank or write 'unknown' to skip DL-M2 for
                                that subject only

Molecular subtype choices (molecular_subtype column or --mol_subtype flag):
  KIAA1549_BRAF  BRAF_V600E  NF1  FGFR  RTK  IDH  MYB  other_MAPK  wildtype  unknown

Usage
-----
  # molecular_subtype column in the spreadsheet (recommended for multi-subject runs):
  python run_full_inference.py \\
      --image_dir     /path/to/T2w_nifti_files \\
      --clinical_xlsx /path/to/Clinical_Features.xlsx \\
      --output_dir    results/report

  # override / fallback for all subjects via CLI:
  python run_full_inference.py \\
      --image_dir     /path/to/T2w_nifti_files \\
      --clinical_xlsx /path/to/Clinical_Features.xlsx \\
      --mol_subtype   KIAA1549_BRAF \\
      --output_dir    results/report

Outputs
-------
  ResNet_Features.xlsx            extracted imaging features
  DLM1_risk_results.csv           DL-M1 risk score + 95% CI + group
  DLM2_risk_results.csv           DL-M2 fusion risk score + 95% CI + group
  DLM1_treatment_scenarios.csv    4 scenarios × 3 time-points (point + CI)
  DLM2_treatment_scenarios.csv    same for DL-M2
  <SubjectID>_DLM1_scenarios.png  survival curves with CI bands (DL-M1)
  <SubjectID>_DLM2_scenarios.png  survival curves with CI bands (DL-M2)
  full_report.csv                 all results in one table
"""

import os
import re
import sys
import types
import pickle
import shutil
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ── Repo layout ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
SEG_DIR   = REPO_ROOT / "05_Segmentation"
DLM1_DIR  = REPO_ROOT / "01_DLM1_Clinico_ResNet"
MOL_DIR   = REPO_ROOT / "02_Molecular_Subtype"
DLM2_DIR  = REPO_ROOT / "03_Late_Fusion_DLM2" / "final_model"

sys.path.insert(0, str(SEG_DIR))
sys.path.insert(0, str(DLM1_DIR))

import model as seg_model_utils
import inference as dlm1_inference
from dataset import CombinedSegmentationDataset

# ── Molecular subtype → multi-hot columns (mol_ prefix) ───────────────────────
# Matches the encoding in 02_Molecular_Subtype/1_Data_Wrangling.ipynb.
# Each alteration is an independent binary flag; co-driver tumors fire multiple.
MOL_TOKEN_MAP = {
    "KIAA1549_BRAF": "mol_KIAA1549_BRAF",
    "BRAF_V600E":    "mol_BRAF_V600E",
    "NF1":           "mol_NF1",
    "FGFR":          "mol_FGFR",
    "RTK":           "mol_RTK",
    "IDH":           "mol_IDH",
    "MYB":           "mol_MYB",
    "other_MAPK":    "mol_other_MAPK",
    "CDKN2A_B":      "mol_CDKN2A_B",
    "wildtype":      None,   # all-zeros row; no column fires
    "unknown":       None,
}
MOL_ALTERATION_COLS = [
    "mol_KIAA1549_BRAF", "mol_BRAF_V600E", "mol_NF1", "mol_FGFR",
    "mol_RTK", "mol_IDH", "mol_MYB", "mol_other_MAPK", "mol_CDKN2A_B",
]
# Clinical columns expected by the molecular model (same encoding as DL-M1 except
# no Tumor Location; Age scaled by 02_Molecular_Subtype/scaler_clinical.pkl).
MOL_CLINICAL_COLS = [
    "Sex", "Age at Diagnosis", "Extent of Tumor Resection", "Chemotherapy", "Radiation",
]
MOL_EXTENT_NORMALIZER = 3
# All 14 features the molecular estimator expects (order must match training):
MOL_FEATURE_COLS = MOL_CLINICAL_COLS + MOL_ALTERATION_COLS

# ── Treatment scenarios ────────────────────────────────────────────────────────
# resection values are in the /3 scale used by engineer_clinical (EXTENT_NORMALIZER=3)
TREATMENT_SCENARIOS = [
    {"label": "Biopsy only + Chemo: Yes",               "resection": 1/3, "chemo": 1, "color": "#377eb8", "ls": "-"},
    {"label": "Partial resection + Chemo: No",           "resection": 2/3, "chemo": 0, "color": "#4daf4a", "ls": "--"},
    {"label": "Partial resection + Chemo: Yes",          "resection": 2/3, "chemo": 1, "color": "#984ea3", "ls": "--"},
    {"label": "Gross/Near total resection + Chemo: No",  "resection": 3/3, "chemo": 0, "color": "#ff7f00", "ls": "-."},
]

SURV_TIME_POINTS = [12, 36, 60]
SURV_LABELS      = ["1-yr PFS", "3-yr PFS", "5-yr PFS"]
CI_ALPHA         = 0.15  # fill_between transparency for CI bands


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _subject_id_from_filename(path: Path) -> str:
    stem = path.name
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    m = re.match(r"^(C\d+|sub[-_]?\d+)", stem)
    return m.group(1) if m else stem.split("_")[0]


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _eval_surv(surv_fn, t: float) -> float:
    if hasattr(surv_fn, "x") and len(surv_fn.x) > 0:
        t = np.clip(t, surv_fn.x[0], surv_fn.x[-1])
    return float(surv_fn(t))


def _surv_curve(model, x_row, time_points):
    """Survival curve for one subject at given time_points (clips to model's event range)."""
    surv_fn = model.predict_survival_function([x_row])[0]
    return np.array([_eval_surv(surv_fn, t) for t in time_points])


def _paper_time_points(refit_model, x_df, n: int = 100) -> np.ndarray:
    """
    Replicate the paper's time-point sampling: np.linspace(t_min, t_max, 100)
    derived from the refit CoxPH model's survival function for this subject.
    All bootstrap curves use the same grid (per-model clipping handled in _surv_curve).
    """
    fn = refit_model.predict_survival_function(x_df)[0]
    return np.linspace(fn.x[0], fn.x[-1], n)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: ResNet feature extraction
# ─────────────────────────────────────────────────────────────────────────────

class _NIfTIDataset(Dataset):
    EXTENSIONS = (".nii.gz", ".nii")

    def __init__(self, image_dir: str):
        self.records = []
        for p in sorted(Path(image_dir).iterdir()):
            if any(str(p).endswith(e) for e in self.EXTENSIONS):
                self.records.append({"subject_id": _subject_id_from_filename(p), "path": str(p)})
        if not self.records:
            raise RuntimeError(f"No NIfTI files found in: {image_dir}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img = nib.load(self.records[idx]["path"]).get_fdata(dtype=np.float32)
        if img.ndim > 3:
            img = img[..., 0]
        img = CombinedSegmentationDataset._normalize(img)
        return torch.from_numpy(np.expand_dims(img, 0)).float(), idx

    def subject_id(self, idx):
        return self.records[idx]["subject_id"]


def _load_seg_backbone(model_path: str, device: torch.device) -> nn.Module:
    opt = types.SimpleNamespace(
        model="resnet", model_depth=34,
        input_W=128, input_H=128, input_D=128,
        resnet_shortcut="B", no_cuda=(device.type == "cpu"),
        n_seg_classes=1, gpu_id=[0], phase="test",
        pretrain_path=None, new_layer_names=["conv_seg"],
    )
    net, _ = seg_model_utils.generate_model(opt)
    ckpt   = torch.load(model_path, map_location=device)
    sd     = ckpt.get("model_state_dict", ckpt)
    sd     = {k.replace("module.", ""): v for k, v in sd.items()}
    inner  = net.module if isinstance(net, nn.DataParallel) else net
    inner.load_state_dict(sd, strict=True)
    return inner


def step1_extract_features(image_dir: str, model_path: str,
                            output_dir: Path, device: torch.device) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("STEP 1 — ResNet Feature Extraction (layer3 GAP+GMP, 512-dim)")
    print("=" * 65)

    if not model_path:
        model_path = str(SEG_DIR / "final_model" / "final_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Segmentation model not found: {model_path}\n"
            "Run:  git lfs pull"
        )

    dataset  = _NIfTIDataset(image_dir)
    loader   = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    backbone = _load_seg_backbone(model_path, device)
    backbone.to(device)
    backbone.eval()
    print(f"  {len(dataset)} subject(s) | device: {device}")

    collected, all_idx, captured = [], [], {}

    def _hook(m, i, o):
        captured["layer3"] = o.detach().cpu()

    handle = backbone.layer3.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for vols, idx in loader:
                backbone(vols.to(device))
                feat = captured["layer3"]
                gap  = feat.mean(dim=[2, 3, 4]).numpy()
                gmp  = feat.amax(dim=[2, 3, 4]).numpy()
                collected.append(np.concatenate([gap, gmp], axis=1))
                all_idx.extend(idx.tolist())
    finally:
        handle.remove()

    features    = np.concatenate(collected, axis=0)
    subject_ids = [dataset.subject_id(i) for i in all_idx]
    df = pd.DataFrame(features, columns=[f"feature_{i:03d}" for i in range(features.shape[1])])
    df.insert(0, "SubjectID", subject_ids)
    df.to_excel(str(output_dir / "ResNet_Features.xlsx"), index=False)
    print(f"  Extracted {features.shape[1]} features for {len(df)} subject(s) ✓")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DL-M1 risk score + 95% bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def _build_dlm1_cfg(clinical_xlsx: str, output_dir: Path) -> dict:
    clin_fname = os.path.basename(clinical_xlsx)
    dest       = output_dir / clin_fname
    if str(Path(clinical_xlsx).resolve()) != str(dest.resolve()):
        shutil.copy(clinical_xlsx, dest)

    cfg = dict(dlm1_inference.CONFIG)
    cfg.update({
        "input_dir":     str(output_dir.resolve()),
        "artifact_dir":  str(DLM1_DIR),
        "model_dir":     str(DLM1_DIR / "outputs" / "models"),
        "threshold_dir": str(DLM1_DIR / "outputs" / "models"),
        "output_dir":    str(output_dir.resolve()),
        "clinical_xlsx": clin_fname,
        "resnet_xlsx":   "ResNet_Features.xlsx",
        "output_csv":    "DLM1_risk_results.csv",
    })
    return cfg


def read_mol_subtypes(clinical_xlsx: str, cli_mol_subtype: str | None) -> dict:
    """
    Returns {SubjectID: mol_subtype_key} for every subject.

    Priority:
      1. 'molecular_subtype' column in the spreadsheet (per-subject)
      2. --mol_subtype CLI flag (fallback applied to all subjects)
    Blank / NaN / unrecognised spreadsheet values fall back to the CLI flag,
    or to 'unknown' if neither is provided.
    """
    df = pd.read_excel(clinical_xlsx)
    has_col = "molecular_subtype" in df.columns

    if not has_col and cli_mol_subtype is None:
        raise ValueError(
            "No 'molecular_subtype' column in the spreadsheet and --mol_subtype not provided.\n"
            "Add the column to Clinical_Features.xlsx or pass --mol_subtype on the command line."
        )

    result = {}
    for _, row in df.iterrows():
        subj = str(row["SubjectID"])
        if has_col:
            val = str(row.get("molecular_subtype", "")).strip()
            val = val if val in MOL_TOKEN_MAP else (cli_mol_subtype or "unknown")
        else:
            val = cli_mol_subtype
        result[subj] = val
    return result


def step2_dlm1(clinical_xlsx: str, output_dir: Path):
    """
    Returns (results_df, X_all_df, X_all_ordered, X_boot, estimator, bootstrap_models,
             median_threshold, refit_model, refit_threshold).
    """
    print("\n" + "=" * 65)
    print("STEP 2 — DL-M1 Risk Score + 95% Bootstrap CI")
    print("=" * 65)

    cfg   = _build_dlm1_cfg(clinical_xlsx, output_dir)
    paths = dlm1_inference.build_paths(cfg)

    encoder         = _load_pkl(paths["encoder_pkl"])
    scaler_clinical = _load_pkl(paths["scaler_clinical_pkl"])
    imputer         = _load_pkl(paths["imputer_pkl"])
    scaler_radiomic = _load_pkl(paths["scaler_radiomic_pkl"])
    remover         = _load_pkl(paths["remover_pkl"])
    dropper         = _load_pkl(paths["dropper_pkl"])
    estimator       = _load_pkl(paths["estimator_pkl"])
    thr_obj         = _load_pkl(paths["risk_threshold_pkl"])
    median_threshold = thr_obj["median_threshold"] if isinstance(thr_obj, dict) else float(thr_obj)

    # Unregularized CoxPH refit on full discovery cohort (12 LASSO features).
    # Used for treatment scenario point estimates so that point estimates and
    # bootstrap CIs come from the same model family (both CoxPH, alpha=0).
    refit_path = DLM1_DIR / "outputs" / "models" / "refit_coxph_model.pkl"
    thr2_path  = DLM1_DIR / "outputs" / "models" / "refit_threshold.pkl"
    refit_model     = _load_pkl(refit_path)
    refit_thr_obj   = _load_pkl(thr2_path)
    refit_threshold = refit_thr_obj["median_threshold"] if isinstance(refit_thr_obj, dict) else float(refit_thr_obj)

    # Load 1000 bootstrap CoxPH models for CI
    boot_path = DLM1_DIR / "outputs" / "models" / "bootstrap_models.pkl"
    bootstrap_models = _load_pkl(boot_path)
    print(f"  Loaded {len(bootstrap_models)} bootstrap models for 95% CI")

    clin_raw   = dlm1_inference.wrangle_clinical(paths["clinical_xlsx"])
    resnet_raw = dlm1_inference.wrangle_resnet(paths["resnet_xlsx"], set(clin_raw.index))
    X_clin     = dlm1_inference.engineer_clinical(clin_raw, encoder, scaler_clinical)
    X_resnet   = dlm1_inference.engineer_resnet(resnet_raw, imputer, scaler_radiomic, remover, dropper)

    X_all = X_clin.merge(X_resnet, left_index=True, right_index=True, how="inner")
    if X_all.empty:
        raise RuntimeError("No subjects after merging clinical + ResNet features — check SubjectID alignment.")

    # Main estimator uses all engineered features (Coxnet, 158 features)
    feat_names = getattr(estimator, "feature_names_in_", None)
    if feat_names is not None:
        X_all_ordered = X_all[list(feat_names)]
    else:
        X_all_ordered = X_all

    # Bootstrap CoxPH models use the LASSO-selected feature subset (12 features)
    boot_feat_names = list(bootstrap_models[0].feature_names_in_)
    X_boot = X_all[boot_feat_names]

    # Point estimates (main Coxnet model)
    point_scores = estimator.predict(X_all_ordered)

    # Bootstrap CI on risk score
    boot_scores = np.array([
        m.predict(X_boot) for m in bootstrap_models
    ])  # (1000, N)

    records = []
    for i, subj in enumerate(X_all_ordered.index):
        score = point_scores[i]
        ci_lo = np.percentile(boot_scores[:, i], 2.5)
        ci_hi = np.percentile(boot_scores[:, i], 97.5)
        group = "High" if score > median_threshold else "Low"
        records.append({
            "Risk Score":    round(score, 4),
            "CI Lower":      round(ci_lo, 4),
            "CI Upper":      round(ci_hi, 4),
            "Risk Group":    group,
        })

    results = pd.DataFrame(records, index=pd.Index(X_all_ordered.index, name="SubjectID"))
    results.to_csv(str(output_dir / "DLM1_risk_results.csv"), index=True)

    print(f"\n  DL-M1 results ({len(results)} subject(s)):")
    print(results.to_string())
    return (results, X_all, X_all_ordered, X_boot, estimator, bootstrap_models,
            median_threshold, refit_model, refit_threshold)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: DL-M2 risk score + 95% bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def _build_mol_features_row(
    mol_subtype_key: str,
    clin_row: pd.Series,
    mol_scaler,
) -> pd.DataFrame:
    """
    Build the 14-feature input row for the molecular estimator:
      [Sex, Age at Diagnosis (scaled), Extent of Tumor Resection (/3),
       Chemotherapy, Radiation, mol_KIAA1549_BRAF, ..., mol_CDKN2A_B]

    clin_row: the subject's raw clinical Series (same columns as Clinical_Features.xlsx
              after label-encoding in engineer_clinical).
    mol_scaler: sklearn StandardScaler fit on Age at Diagnosis (molecular cohort).
    """
    row = {}

    # ── Clinical features (label-encoded, same as DL-M1 except no Tumor Location) ──
    sex_raw = clin_row.get("legal_sex", clin_row.get("Sex", 0))
    row["Sex"] = 1 if str(sex_raw).strip().lower() in ("male", "1") else 0

    age_raw = float(clin_row.get("age_at_event_days", clin_row.get("Age at Diagnosis", 0)))
    row["Age at Diagnosis"] = float(mol_scaler.transform([[age_raw]])[0][0])

    resection_map = {
        "not applicable": 0, "unavailable": 0,
        "biopsy only": 1,
        "partial resection": 2,
        "gross/near total resection": 3,
    }
    res_raw = str(clin_row.get("extent_of_tumor_resection",
                               clin_row.get("Extent of Tumor Resection", "unavailable"))).lower()
    row["Extent of Tumor Resection"] = resection_map.get(res_raw, 0) / MOL_EXTENT_NORMALIZER

    chemo_raw = str(clin_row.get("chemotherapy", clin_row.get("Chemotherapy", "No"))).lower()
    row["Chemotherapy"] = 1 if chemo_raw in ("yes", "1") else 0

    rad_raw = str(clin_row.get("radiation", clin_row.get("Radiation", "No"))).lower()
    row["Radiation"] = 1 if rad_raw in ("yes", "1") else 0

    # ── Multi-hot molecular alteration columns ─────────────────────────────────
    for col in MOL_ALTERATION_COLS:
        row[col] = 0
    col = MOL_TOKEN_MAP.get(mol_subtype_key)
    if col is not None:
        row[col] = 1

    return pd.DataFrame([row], columns=MOL_FEATURE_COLS)


def step3_dlm2(
    dlm1_results: pd.DataFrame,
    mol_subtypes: dict,
    clin_df: pd.DataFrame,
    output_dir: Path,
):
    """
    mol_subtypes: {SubjectID: mol_subtype_key} — per-subject, from read_mol_subtypes().
    clin_df: raw clinical DataFrame indexed by SubjectID (for clinical features).
    Subjects whose subtype is 'unknown' are skipped (no DL-M2 row produced).

    Returns (results_df, mol_risk_by_subj, fusion_est, scalers, bootstrap_entries, threshold).
    """
    print("\n" + "=" * 65)
    print("STEP 3 — DL-M2 Fusion Risk Score + 95% Bootstrap CI")
    print("=" * 65)

    mol_est      = _load_pkl(MOL_DIR / "outputs" / "models" / "estimator.pkl")
    mol_scaler   = _load_pkl(MOL_DIR / "scaler_clinical.pkl")
    fusion_est   = _load_pkl(DLM2_DIR / "model.pkl")
    scalers      = _load_pkl(DLM2_DIR / "scalers.pkl")
    threshold    = float(open(DLM2_DIR / "threshold.txt").read().split(":")[1].split("\n")[0].strip())
    boot_entries = _load_pkl(DLM2_DIR / "bootstrap_entries.pkl")
    print(f"  Loaded {len(boot_entries)} bootstrap entries for 95% CI")
    print(f"  DL-M2 threshold: {threshold:.6f}")

    records          = {}
    mol_risk_by_subj = {}

    for subj in dlm1_results.index:
        mol_key = mol_subtypes.get(subj, "unknown")
        if mol_key == "unknown":
            print(f"  {subj}: molecular_subtype = unknown — skipping DL-M2")
            continue

        clin_row  = clin_df.loc[subj] if subj in clin_df.index else pd.Series(dtype=object)
        cr_risk   = float(dlm1_results.loc[subj, "Risk Score"])
        X_mol_row = _build_mol_features_row(mol_key, clin_row, mol_scaler)
        mol_risk  = float(mol_est.predict(X_mol_row)[0])

        # Point estimate
        cr_sc  = scalers["Clinical-ResNet"].transform([[cr_risk]])[0][0]
        mol_sc = scalers["Molecular"].transform([[mol_risk]])[0][0]
        score  = float(fusion_est.predict(np.array([[cr_sc, mol_sc]]))[0])

        # Bootstrap CI
        boot_scores = np.array([
            entry["model"].predict(np.array([[
                entry["scaler_cr"].transform([[cr_risk]])[0][0],
                entry["scaler_mol"].transform([[mol_risk]])[0][0],
            ]]))[0]
            for entry in boot_entries
        ])

        ci_lo = float(np.percentile(boot_scores, 2.5))
        ci_hi = float(np.percentile(boot_scores, 97.5))

        mol_risk_by_subj[subj] = mol_risk
        records[subj] = {
            "Molecular_Subtype":    mol_key,
            "DLM1_Risk_Score":      round(cr_risk, 4),
            "Mol_Risk_Score":       round(mol_risk, 4),
            "DLM2_Risk_Score":      round(score, 4),
            "DLM2_CI_Lower":        round(ci_lo, 4),
            "DLM2_CI_Upper":        round(ci_hi, 4),
            "DLM2_Risk_Group":      "High" if score > threshold else "Low",
        }
        print(f"  {subj}: subtype={mol_key}  DL-M2={score:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  {records[subj]['DLM2_Risk_Group']}")

    if not records:
        print("  No subjects with known molecular subtype — DL-M2 skipped for all.")
        return None, {}, fusion_est, scalers, boot_entries, threshold

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "SubjectID"
    df.to_csv(str(output_dir / "DLM2_risk_results.csv"))
    print(f"\n  Saved → {output_dir / 'DLM2_risk_results.csv'}")
    return df, mol_risk_by_subj, fusion_est, scalers, boot_entries, threshold


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4a: DL-M1 treatment scenarios + 95% CI
# ─────────────────────────────────────────────────────────────────────────────

def step4a_dlm1_scenarios(X_boot: pd.DataFrame,
                           refit_model, bootstrap_models: list,
                           refit_threshold: float,
                           output_dir: Path) -> pd.DataFrame:
    """
    Point estimates and 95% CI both come from CoxPHSurvivalAnalysis(alpha=0) on the
    12 LASSO-selected features, matching the paper's treatment_scenario_DLM1.py:
      - refit_model:      unregularized CoxPH fit on full discovery cohort
      - bootstrap_models: 1000 unregularized CoxPH models on bootstrap resamples
    This guarantees the point estimate always falls within its own CI band.
    """
    print("\n" + "=" * 65)
    print("STEP 4a — DL-M1 Treatment Scenarios + 95% CI")
    print("=" * 65)

    rows = []

    for subject_id in X_boot.index:
        x_boot = X_boot.loc[subject_id]   # 12 LASSO-selected features

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax1, ax2 = axes
        scenario_table = []

        for sc in TREATMENT_SCENARIOS:
            # Patch treatment features in the 12-feature subset (used by both refit and bootstrap)
            x_sc = x_boot.copy()
            x_sc["Extent of Tumor Resection"] = sc["resection"]
            x_sc["Chemotherapy"]              = sc["chemo"]
            x_df = pd.DataFrame([x_sc], columns=list(x_sc.index))

            # Time grid: linspace from refit model's event range (paper's exact approach).
            # All 1000 bootstrap models use the same grid; each clips to its own event range.
            time_points = _paper_time_points(refit_model, x_df)

            # Point estimate: unregularized CoxPH refit on full discovery cohort
            pt_curve = _surv_curve(refit_model, x_sc.values, time_points)
            pt_risk  = float(refit_model.predict(x_df)[0])

            # Bootstrap CI: 1000 unregularized CoxPH models (same family, same 12 features)
            # → point estimate is guaranteed to fall within its own CI band
            boot_curves = np.array([
                _surv_curve(m, x_sc.values, time_points)
                for m in bootstrap_models
            ])
            boot_risks  = np.array([float(m.predict(x_df)[0]) for m in bootstrap_models])
            ci_lo = np.percentile(boot_curves, 2.5,  axis=0)
            ci_hi = np.percentile(boot_curves, 97.5, axis=0)
            risk_ci_lo = float(np.percentile(boot_risks, 2.5))
            risk_ci_hi = float(np.percentile(boot_risks, 97.5))
            risk_group = "High" if pt_risk > refit_threshold else "Low"

            # PFS at 1/3/5 yr: nearest linspace index (paper's exact lookup)
            def _pfs_at(t_target, curve):
                idx = int(np.argmin(np.abs(time_points - t_target)))
                return float(curve[idx])

            pfs    = [_pfs_at(t, pt_curve)                                          for t in SURV_TIME_POINTS]
            pfs_lo = [float(np.percentile([_pfs_at(t, bc) for bc in boot_curves], 2.5))  for t in SURV_TIME_POINTS]
            pfs_hi = [float(np.percentile([_pfs_at(t, bc) for bc in boot_curves], 97.5)) for t in SURV_TIME_POINTS]

            rows.append({
                "SubjectID":  subject_id,
                "Model":      "DL-M1",
                "Scenario":   sc["label"],
                "Risk Score": round(pt_risk, 4),
                "Risk CI":    f"[{risk_ci_lo:.4f}, {risk_ci_hi:.4f}]",
                "Risk Group": risk_group,
                **{SURV_LABELS[i]: round(pfs[i], 3)    for i in range(3)},
                **{f"{SURV_LABELS[i]}_CI": f"[{pfs_lo[i]:.3f}, {pfs_hi[i]:.3f}]" for i in range(3)},
            })
            scenario_table.append({
                "label": sc["label"],
                "pt_risk": pt_risk, "risk_ci_lo": risk_ci_lo, "risk_ci_hi": risk_ci_hi,
                "risk_group": risk_group,
                "pfs": pfs, "pfs_lo": pfs_lo, "pfs_hi": pfs_hi,
            })

            ax1.plot(time_points, pt_curve,
                     color=sc["color"], linestyle=sc["ls"],
                     linewidth=2.5, label=sc["label"], alpha=0.9)
            ax1.fill_between(time_points, ci_lo, ci_hi,
                             color=sc["color"], alpha=CI_ALPHA)

        ax1.set_xlabel("Time (months)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Progression-Free Survival Probability", fontsize=12, fontweight="bold")
        ax1.set_title(f"Subject {subject_id}: Survival Under Treatment Scenarios",
                      fontsize=13, fontweight="bold")
        ax1.legend(loc="lower left", fontsize=9, framealpha=0.95)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(left=0)
        ax1.grid(alpha=0.3)

        # ── Summary table ────────────────────────────────────────────────────
        ax2.axis("off")
        headers = ["Treatment Scenario", "Risk Score\n(95% CI)", "Risk\nGroup"] + \
                  [f"{l}\n(95% CI)" for l in SURV_LABELS]
        table_data = []
        for st in scenario_table:
            row_td = [
                st["label"],
                f"{st['pt_risk']:.2f}\n[{st['risk_ci_lo']:.2f}, {st['risk_ci_hi']:.2f}]",
                st["risk_group"],
            ]
            for i in range(3):
                row_td.append(f"{st['pfs'][i]:.2f}\n[{st['pfs_lo'][i]:.2f}, {st['pfs_hi'][i]:.2f}]")
            table_data.append(row_td)
        table = ax2.table(cellText=table_data, colLabels=headers,
                          cellLoc="center", loc="center",
                          colWidths=[0.32, 0.15, 0.10] + [0.14] * 3)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 3.2)
        for j in range(len(headers)):
            table[(0, j)].set_facecolor("#1a9850")
            table[(0, j)].set_text_props(weight="bold", color="white")
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(len(headers)):
                    table[(i, j)].set_facecolor("#f0f0f0")
            rg = table_data[i - 1][2]
            table[(i, 2)].set_facecolor("#ffcccc" if rg == "High" else "#ccffcc")
        ax2.set_title("Treatment Scenario Summary\n"
                      "Point estimate: refit CoxPH (discovery); 95% CI: 2.5th–97.5th bootstrap percentile",
                      fontsize=9, fontweight="bold", pad=20)

        fig.suptitle(f"DL-M1 Survival Analysis — {subject_id}",
                     fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig.savefig(str(output_dir / f"{subject_id}_DLM1_scenarios.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  {subject_id}: DL-M1 scenario plot saved")

    df = pd.DataFrame(rows)
    df.to_csv(str(output_dir / "DLM1_treatment_scenarios.csv"), index=False)
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4b: DL-M2 treatment scenarios + 95% CI
# ─────────────────────────────────────────────────────────────────────────────

def _dlm2_fusion_curve(entry_or_tuple, cr_raw, mol_raw, time_points):
    """Predict survival curve from a fusion model given raw (unscaled) risk scores."""
    if isinstance(entry_or_tuple, dict):
        model     = entry_or_tuple["model"]
        scaler_cr = entry_or_tuple["scaler_cr"]
        scaler_mol= entry_or_tuple["scaler_mol"]
    else:
        model, scaler_cr, scaler_mol = entry_or_tuple

    cr_sc  = scaler_cr.transform([[cr_raw]])[0][0]
    mol_sc = scaler_mol.transform([[mol_raw]])[0][0]
    surv_fn = model.predict_survival_function(np.array([[cr_sc, mol_sc]]))[0]
    return np.array([_eval_surv(surv_fn, t) for t in time_points])


def step4b_dlm2_scenarios(X_all_ordered: pd.DataFrame,
                           dlm1_estimator,
                           mol_risk_by_subj: dict,
                           fusion_est,
                           scalers: dict,
                           bootstrap_entries: list,
                           dlm1_results: pd.DataFrame,
                           dlm2_threshold: float,
                           output_dir: Path) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("STEP 4b — DL-M2 Treatment Scenarios + 95% CI")
    print("=" * 65)

    # Wrap point-estimate fusion model as a pseudo-entry using discovery scalers
    point_entry = {
        "model":      fusion_est,
        "scaler_cr":  scalers["Clinical-ResNet"],
        "scaler_mol": scalers["Molecular"],
    }

    time_points = np.arange(0, 61, 1)
    rows = []
    for subject_id in X_all_ordered.index:
        if subject_id not in mol_risk_by_subj:
            print(f"  {subject_id}: no molecular subtype — skipping DL-M2 scenario plot")
            continue
        x_row        = X_all_ordered.loc[subject_id]
        mol_raw_subj = mol_risk_by_subj[subject_id]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax1, ax2 = axes
        scenario_table = []

        for sc in TREATMENT_SCENARIOS:
            x_sc = x_row.copy()
            x_sc["Extent of Tumor Resection"] = sc["resection"]
            x_sc["Chemotherapy"]              = sc["chemo"]

            # DL-M1 risk score for this scenario (point estimate)
            cr_raw_sc = float(dlm1_estimator.predict([x_sc.values])[0])

            # Point estimate fusion risk score and survival curve
            cr_sc  = scalers["Clinical-ResNet"].transform([[cr_raw_sc]])[0][0]
            mol_sc = scalers["Molecular"].transform([[mol_raw_subj]])[0][0]
            pt_risk = float(fusion_est.predict(np.array([[cr_sc, mol_sc]]))[0])
            pt_curve = _dlm2_fusion_curve(point_entry, cr_raw_sc, mol_raw_subj, time_points)

            # Bootstrap CI — each entry re-scales cr_raw and mol_raw with its own scalers
            boot_curves = np.array([
                _dlm2_fusion_curve(entry, cr_raw_sc, mol_raw_subj, time_points)
                for entry in bootstrap_entries
            ])
            boot_risks = np.array([
                float(e["model"].predict(np.array([[
                    e["scaler_cr"].transform([[cr_raw_sc]])[0][0],
                    e["scaler_mol"].transform([[mol_raw_subj]])[0][0],
                ]]))[0])
                for e in bootstrap_entries
            ])

            ci_lo = np.percentile(boot_curves, 2.5,  axis=0)
            ci_hi = np.percentile(boot_curves, 97.5, axis=0)
            risk_ci_lo = float(np.percentile(boot_risks, 2.5))
            risk_ci_hi = float(np.percentile(boot_risks, 97.5))
            risk_group = "High" if pt_risk > dlm2_threshold else "Low"

            pfs    = [float(pt_curve[t])                             for t in SURV_TIME_POINTS]
            pfs_lo = [float(np.percentile(boot_curves[:, t], 2.5))  for t in SURV_TIME_POINTS]
            pfs_hi = [float(np.percentile(boot_curves[:, t], 97.5)) for t in SURV_TIME_POINTS]

            rows.append({
                "SubjectID":  subject_id,
                "Model":      "DL-M2",
                "Scenario":   sc["label"],
                "Risk Score": round(pt_risk, 4),
                "Risk CI":    f"[{risk_ci_lo:.4f}, {risk_ci_hi:.4f}]",
                "Risk Group": risk_group,
                **{SURV_LABELS[i]: round(pfs[i], 3)    for i in range(3)},
                **{f"{SURV_LABELS[i]}_CI": f"[{pfs_lo[i]:.3f}, {pfs_hi[i]:.3f}]" for i in range(3)},
            })
            scenario_table.append({
                "label": sc["label"],
                "pt_risk": pt_risk, "risk_ci_lo": risk_ci_lo, "risk_ci_hi": risk_ci_hi,
                "risk_group": risk_group,
                "pfs": pfs, "pfs_lo": pfs_lo, "pfs_hi": pfs_hi,
            })

            ax1.plot(time_points, pt_curve,
                     color=sc["color"], linestyle=sc["ls"],
                     linewidth=2.5, label=sc["label"], alpha=0.9)
            ax1.fill_between(time_points, ci_lo, ci_hi,
                             color=sc["color"], alpha=CI_ALPHA)

        ax1.set_xlabel("Time (months)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Progression-Free Survival Probability", fontsize=12, fontweight="bold")
        ax1.set_title(f"Subject {subject_id}: Survival Under Treatment Scenarios",
                      fontsize=13, fontweight="bold")
        ax1.legend(loc="lower left", fontsize=9, framealpha=0.95)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(0, time_points[-1])
        ax1.grid(alpha=0.3)

        # ── Summary table ────────────────────────────────────────────────────
        ax2.axis("off")
        headers = ["Treatment Scenario", "Risk Score\n(95% CI)", "Risk\nGroup"] + \
                  [f"{l}\n(95% CI)" for l in SURV_LABELS]
        table_data = []
        for st in scenario_table:
            row_td = [
                st["label"],
                f"{st['pt_risk']:.2f}\n[{st['risk_ci_lo']:.2f}, {st['risk_ci_hi']:.2f}]",
                st["risk_group"],
            ]
            for i in range(3):
                row_td.append(f"{st['pfs'][i]:.2f}\n[{st['pfs_lo'][i]:.2f}, {st['pfs_hi'][i]:.2f}]")
            table_data.append(row_td)
        table = ax2.table(cellText=table_data, colLabels=headers,
                          cellLoc="center", loc="center",
                          colWidths=[0.32, 0.15, 0.10] + [0.14] * 3)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 3.2)
        for j in range(len(headers)):
            table[(0, j)].set_facecolor("#1a9850")
            table[(0, j)].set_text_props(weight="bold", color="white")
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(len(headers)):
                    table[(i, j)].set_facecolor("#f0f0f0")
            rg = table_data[i - 1][2]
            table[(i, 2)].set_facecolor("#ffcccc" if rg == "High" else "#ccffcc")
        ax2.set_title("Treatment Scenario Summary\n"
                      "Point estimate: discovery fusion model; 95% CI: 2.5th–97.5th bootstrap percentile",
                      fontsize=9, fontweight="bold", pad=20)

        fig.suptitle(f"DL-M2 Survival Analysis — {subject_id}",
                     fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig.savefig(str(output_dir / f"{subject_id}_DLM2_scenarios.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  {subject_id}: DL-M2 scenario plot saved")

    df = pd.DataFrame(rows)
    df.to_csv(str(output_dir / "DLM2_treatment_scenarios.csv"), index=False)
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Combine all results into one summary table
# ─────────────────────────────────────────────────────────────────────────────

def combine_results(dlm1: pd.DataFrame, dlm2: pd.DataFrame | None,
                    sc_m1: pd.DataFrame, sc_m2: pd.DataFrame | None,
                    output_dir: Path) -> pd.DataFrame:
    summary = dlm1.rename(columns={
        "Risk Score": "DLM1_Score",
        "CI Lower":   "DLM1_CI_Lo",
        "CI Upper":   "DLM1_CI_Hi",
        "Risk Group": "DLM1_Group",
    })
    if dlm2 is not None:
        summary = summary.join(dlm2[["DLM2_Risk_Score", "DLM2_CI_Lower", "DLM2_CI_Upper", "DLM2_Risk_Group"]])

    # Best-case (GTR no chemo) PFS from each model
    for model_label, sc_df in [("M1", sc_m1), ("M2", sc_m2)]:
        if sc_df is None:
            continue
        best = sc_df[sc_df["Scenario"].str.contains("Gross")].set_index("SubjectID")
        for lbl in SURV_LABELS:
            summary[f"GTR_{model_label}_{lbl}"]    = best.get(lbl,    np.nan)
            summary[f"GTR_{model_label}_{lbl}_CI"] = best.get(f"{lbl}_CI", "")

    summary.to_csv(str(output_dir / "full_report.csv"))
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full pLGG Risk Inference with 95% Bootstrap CI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Molecular subtype per subject — add a 'molecular_subtype' column to Clinical_Features.xlsx:
  KIAA1549_BRAF  BRAF_V600E  NF1  FGFR  RTK  IDH  MYB  other_MAPK  wildtype  unknown

--mol_subtype applies the same subtype to all subjects (fallback when the column is absent).

Examples:
  # molecular_subtype column in spreadsheet (recommended):
  python run_full_inference.py \\
      --image_dir     example/T2w_scans \\
      --clinical_xlsx example/Clinical_Features.xlsx \\
      --output_dir    results/report

  # single subtype for all subjects via CLI:
  python run_full_inference.py \\
      --image_dir     example/T2w_scans \\
      --clinical_xlsx example/Clinical_Features.xlsx \\
      --mol_subtype   KIAA1549_BRAF \\
      --output_dir    results/report
""",
    )
    parser.add_argument("--image_dir",     required=True)
    parser.add_argument("--clinical_xlsx", required=True)
    parser.add_argument("--mol_subtype",   default=None, choices=list(MOL_SUBTYPE_MAP.keys()),
                        help="Fallback subtype for all subjects when 'molecular_subtype' "
                             "column is absent from the spreadsheet (default: read from spreadsheet)")
    parser.add_argument("--output_dir",    default="inference_report")
    parser.add_argument("--model_path",    default=None,
                        help="Path to .pth (default: 05_Segmentation/final_model/final_model.pth)")
    parser.add_argument("--skip_step1",    action="store_true",
                        help="Skip feature extraction (ResNet_Features.xlsx already in output_dir)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 65)
    print("pLGG Full Risk Inference Pipeline")
    print("=" * 65)

    # Resolve molecular subtypes per subject (spreadsheet column takes priority)
    mol_subtypes = read_mol_subtypes(args.clinical_xlsx, args.mol_subtype)
    has_any_mol  = any(v != "unknown" for v in mol_subtypes.values())

    if not args.skip_step1:
        step1_extract_features(args.image_dir, args.model_path, output_dir, device)

    (dlm1_results, X_all, X_all_ord, X_boot, estimator, boot_models,
     dlm1_threshold, refit_model, refit_threshold) = \
        step2_dlm1(args.clinical_xlsx, output_dir)

    dlm2_results = mol_risk_by_subj = fusion_est = scalers = boot_entries = None
    dlm2_threshold = None
    if has_any_mol:
        clin_df = pd.read_excel(args.clinical_xlsx).set_index("SubjectID")
        dlm2_results, mol_risk_by_subj, fusion_est, scalers, boot_entries, dlm2_threshold = \
            step3_dlm2(dlm1_results, mol_subtypes, clin_df, output_dir)
    else:
        print("\nSTEP 3 — Skipped (all subjects have molecular_subtype = unknown)")

    sc_m1 = step4a_dlm1_scenarios(X_boot, refit_model, boot_models, refit_threshold, output_dir)

    sc_m2 = None
    if dlm2_results is not None and mol_risk_by_subj:
        sc_m2 = step4b_dlm2_scenarios(
            X_all_ord, estimator, mol_risk_by_subj,
            fusion_est, scalers, boot_entries, dlm1_results, dlm2_threshold, output_dir
        )

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    summary = combine_results(dlm1_results, dlm2_results, sc_m1, sc_m2, output_dir)
    print(summary.to_string())
    print(f"\n  All results saved to: {output_dir.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
