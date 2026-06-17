#!/usr/bin/env python3
"""
run_full_inference.py — Full pLGG Risk Inference Pipeline
==========================================================
Given a T2w MRI volume and clinical information for one or more new subjects,
this script produces a complete risk report in four steps:

  Step 1 — ResNet feature extraction
             T2w NIfTI  →  layer3 GAP+GMP features (512-dim)

  Step 2 — DL-M1 risk score and group
             Clinical + ResNet features  →  penalized Cox model
             Output: Risk Score, Risk Group (High / Low)

  Step 3 — DL-M2 risk score and group  [requires --mol_subtype]
             DL-M1 score + Molecular subtype score  →  late-fusion Cox model
             Output: Fusion Risk Score, Fusion Risk Group (High / Low)

  Step 4 — Treatment scenario simulations
             4 resection × chemotherapy combinations  →  survival curves + table

Usage
-----
    python run_full_inference.py \\
        --image_dir     /path/to/T2w_nifti_files \\
        --clinical_xlsx /path/to/Clinical_Features.xlsx \\
        --mol_subtype   KIAA1549_BRAF \\
        --output_dir    /path/to/report

    # Skip DL-M2 (molecular subtype unknown):
    python run_full_inference.py \\
        --image_dir     /path/to/T2w_nifti_files \\
        --clinical_xlsx /path/to/Clinical_Features.xlsx \\
        --mol_subtype   unknown \\
        --output_dir    /path/to/report

Molecular subtype choices (--mol_subtype):
    KIAA1549_BRAF   BRAF_V600E   NF1   FGFR   RTK
    IDH   MYB   other_MAPK   wildtype   unknown

Outputs written to --output_dir:
    ResNet_Features.xlsx          — extracted imaging features
    DLM1_risk_results.csv         — DL-M1 risk scores and groups
    DLM2_risk_results.csv         — DL-M2 fusion risk scores/groups (if mol_subtype given)
    treatment_scenarios.csv       — predicted 1/3/5-yr PFS per scenario
    treatment_scenarios.png       — survival curve plot per scenario
    full_report.csv               — all results combined in one table
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

# Add segmentation and DLM1 dirs to path so their modules are importable
sys.path.insert(0, str(SEG_DIR))
sys.path.insert(0, str(DLM1_DIR))

import model as seg_model_utils
import inference as dlm1_inference
from dataset import CombinedSegmentationDataset

# ── Molecular subtype → one-hot column ────────────────────────────────────────
MOL_SUBTYPE_MAP = {
    "KIAA1549_BRAF": "mol_group_KIAA1549_BRAF",
    "BRAF_V600E":    "mol_group_BRAF_V600E",
    "NF1":           "mol_group_NF1",
    "FGFR":          "mol_group_FGFR",
    "RTK":           "mol_group_RTK",
    "IDH":           "mol_group_IDH",
    "MYB":           "mol_group_MYB",
    "other_MAPK":    "mol_group_other_MAPK",
    "wildtype":      "mol_group_other",
    "unknown":       None,
}

# ── Treatment scenarios ────────────────────────────────────────────────────────
# resection values are on the /3 scale used by DL-M1's engineer_clinical()
# (EXTENT_NORMALIZER=3 → biopsy=1/3, partial=2/3, GTR=3/3=1.0)
TREATMENT_SCENARIOS = [
    {"label": "Biopsy only + Chemo: Yes",               "resection": 1/3, "chemo": 1, "color": "#377eb8", "ls": "-"},
    {"label": "Partial resection + Chemo: No",           "resection": 2/3, "chemo": 0, "color": "#4daf4a", "ls": "--"},
    {"label": "Partial resection + Chemo: Yes",          "resection": 2/3, "chemo": 1, "color": "#984ea3", "ls": "--"},
    {"label": "Gross/Near total resection + Chemo: No",  "resection": 3/3, "chemo": 0, "color": "#ff7f00", "ls": "-."},
]

SURV_TIME_POINTS = [12, 36, 60]
SURV_LABELS      = ["1-yr PFS", "3-yr PFS", "5-yr PFS"]


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


def _load_pkl(path, label):
    with open(path, "rb") as f:
        return pickle.load(f)


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
    """Load the fine-tuned ResNet34; returns the unwrapped backbone (no DataParallel)."""
    opt = types.SimpleNamespace(
        model="resnet", model_depth=34,
        input_W=128, input_H=128, input_D=128,
        resnet_shortcut="B",
        no_cuda=(device.type == "cpu"),
        n_seg_classes=2, gpu_id=[0],
        phase="test", pretrain_path=None,
        new_layer_names=["conv_seg"],
    )
    net, _ = seg_model_utils.generate_model(opt)
    ckpt = torch.load(model_path, map_location=device)
    sd   = ckpt.get("model_state_dict", ckpt)
    sd   = {k.replace("module.", ""): v for k, v in sd.items()}
    # generate_model wraps in DataParallel only when CUDA is used
    inner = net.module if isinstance(net, nn.DataParallel) else net
    inner.load_state_dict(sd, strict=True)
    return inner


def step1_extract_features(image_dir: str, model_path: str,
                            output_dir: Path, device: torch.device) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 1 — ResNet Feature Extraction (layer3 GAP+GMP)")
    print("=" * 60)

    if not model_path:
        model_path = str(SEG_DIR / "final_model" / "final_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Segmentation model not found: {model_path}\n"
            "Run:  git lfs pull   (to download final_model.pth)"
        )

    dataset = _NIfTIDataset(image_dir)
    loader  = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    print(f"  Found {len(dataset)} subject(s) in {image_dir}")
    backbone = _load_seg_backbone(model_path, device)
    backbone.to(device)
    backbone.eval()
    print("  Segmentation backbone loaded ✓")

    collected, all_idx, captured = [], [], {}

    def _hook(m, i, o):
        captured["layer3"] = o.detach().cpu()

    handle = backbone.layer3.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for vols, idx in loader:
                backbone(vols.to(device))
                feat = captured["layer3"]                # (B, 256, D, H, W)
                gap  = feat.mean(dim=[2, 3, 4]).numpy()  # (B, 256)
                gmp  = feat.amax(dim=[2, 3, 4]).numpy()  # (B, 256)
                collected.append(np.concatenate([gap, gmp], axis=1))
                all_idx.extend(idx.tolist())
    finally:
        handle.remove()

    features    = np.concatenate(collected, axis=0)
    subject_ids = [dataset.subject_id(i) for i in all_idx]
    feat_cols   = [f"feature_{i:03d}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feat_cols)
    df.insert(0, "SubjectID", subject_ids)

    out_path = output_dir / "ResNet_Features.xlsx"
    df.to_excel(str(out_path), index=False)
    print(f"  Extracted {features.shape[1]} features for {len(df)} subject(s)")
    print(f"  Saved → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DL-M1 risk score  +  return X_all for reuse in step 4
# ─────────────────────────────────────────────────────────────────────────────

def step2_dlm1(clinical_xlsx: str, output_dir: Path):
    """
    Returns (results_df, X_all_df, estimator, median_threshold).
    X_all_df is the fully engineered feature matrix used by DL-M1.
    """
    print("\n" + "=" * 60)
    print("STEP 2 — DL-M1 Risk Score (Clinical-ResNet Cox Model)")
    print("=" * 60)

    clin_fname = os.path.basename(clinical_xlsx)
    dest       = output_dir / clin_fname
    if str(Path(clinical_xlsx).resolve()) != str(dest.resolve()):
        shutil.copy(clinical_xlsx, dest)

    cfg = dict(dlm1_inference.CONFIG)
    cfg.update({
        "input_dir":     str(output_dir),
        "artifact_dir":  str(DLM1_DIR),
        "model_dir":     str(DLM1_DIR / "outputs" / "models"),
        "threshold_dir": str(DLM1_DIR / "outputs" / "models"),
        "output_dir":    str(output_dir),
        "clinical_xlsx": clin_fname,
        "resnet_xlsx":   "ResNet_Features.xlsx",
        "output_csv":    "DLM1_risk_results.csv",
    })

    paths = dlm1_inference.build_paths(cfg)

    encoder         = _load_pkl(paths["encoder_pkl"],         "encoder")
    scaler_clinical = _load_pkl(paths["scaler_clinical_pkl"], "scaler_clinical")
    imputer         = _load_pkl(paths["imputer_pkl"],         "imputer")
    scaler_radiomic = _load_pkl(paths["scaler_radiomic_pkl"], "scaler_radiomic")
    remover         = _load_pkl(paths["remover_pkl"],         "remover")
    dropper         = _load_pkl(paths["dropper_pkl"],         "dropper")
    estimator       = _load_pkl(paths["estimator_pkl"],       "estimator")
    thr_obj         = _load_pkl(paths["risk_threshold_pkl"],  "threshold")
    median_threshold = thr_obj["median_threshold"] if isinstance(thr_obj, dict) else float(thr_obj)

    clin_raw   = dlm1_inference.wrangle_clinical(paths["clinical_xlsx"])
    resnet_raw = dlm1_inference.wrangle_resnet(paths["resnet_xlsx"], set(clin_raw.index))
    X_clin     = dlm1_inference.engineer_clinical(clin_raw, encoder, scaler_clinical)
    X_resnet   = dlm1_inference.engineer_resnet(resnet_raw, imputer, scaler_radiomic, remover, dropper)

    X_all = X_clin.merge(X_resnet, left_index=True, right_index=True, how="inner")
    if X_all.empty:
        raise RuntimeError("No subjects after merging clinical + ResNet features — check SubjectID alignment.")

    results = dlm1_inference.predict_risk(X_all, estimator, median_threshold)
    results.to_csv(str(output_dir / "DLM1_risk_results.csv"), index=True)

    print(f"\n  DL-M1 results ({len(results)} subject(s)):")
    print(results.to_string())
    print(f"\n  Saved → {output_dir / 'DLM1_risk_results.csv'}")
    return results, X_all, estimator, median_threshold


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: DL-M2 risk score (late fusion)
# ─────────────────────────────────────────────────────────────────────────────

def _build_mol_features(subject_ids: list, mol_subtype_key: str) -> pd.DataFrame:
    all_cols = [v for v in MOL_SUBTYPE_MAP.values() if v is not None]
    df = pd.DataFrame(0, index=subject_ids, columns=all_cols)
    df[MOL_SUBTYPE_MAP[mol_subtype_key]] = 1
    return df


def step3_dlm2(dlm1_results: pd.DataFrame, mol_subtype: str,
               output_dir: Path) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 3 — DL-M2 Risk Score (Late Fusion + Molecular Subtype)")
    print("=" * 60)

    mol_est    = _load_pkl(MOL_DIR / "outputs" / "models" / "estimator.pkl", "mol_estimator")
    fusion_est = _load_pkl(DLM2_DIR / "model.pkl",   "fusion_estimator")
    scalers    = _load_pkl(DLM2_DIR / "scalers.pkl", "scalers")
    thr_line   = open(DLM2_DIR / "threshold.txt").read().split(":")[1].split("\n")[0].strip()
    threshold  = float(thr_line)

    subject_ids   = dlm1_results.index.tolist()
    X_mol         = _build_mol_features(subject_ids, mol_subtype)
    mol_risk      = mol_est.predict(X_mol)
    cr_risk       = dlm1_results["Risk Score"].values

    cr_scaled     = scalers["Clinical-ResNet"].transform(cr_risk.reshape(-1, 1)).flatten()
    mol_scaled    = scalers["Molecular"].transform(mol_risk.reshape(-1, 1)).flatten()
    fusion_scores = fusion_est.predict(np.column_stack([cr_scaled, mol_scaled]))

    df = pd.DataFrame({
        "DLM1_Risk_Score":      cr_risk,
        "Molecular_Risk_Score": mol_risk,
        "DLM2_Risk_Score":      fusion_scores,
        "DLM2_Risk_Group":      np.where(fusion_scores > threshold, "High", "Low"),
    }, index=pd.Index(subject_ids, name="SubjectID"))

    df.to_csv(str(output_dir / "DLM2_risk_results.csv"))
    print(f"  Molecular subtype : {mol_subtype}  →  {MOL_SUBTYPE_MAP[mol_subtype]}")
    print(f"  DL-M2 threshold   : {threshold:.6f}")
    print(f"\n  DL-M2 results ({len(df)} subject(s)):")
    print(df.to_string())
    print(f"\n  Saved → {output_dir / 'DLM2_risk_results.csv'}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Treatment scenario simulations
# ─────────────────────────────────────────────────────────────────────────────

def _eval_surv(surv_fn, t: float) -> float:
    if hasattr(surv_fn, "x") and len(surv_fn.x) > 0:
        t = np.clip(t, surv_fn.x[0], surv_fn.x[-1])
    return float(surv_fn(t))


def step4_treatment_scenarios(X_all: pd.DataFrame, estimator,
                               output_dir: Path) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 4 — Treatment Scenario Simulations")
    print("=" * 60)

    feat_names = getattr(estimator, "feature_names_in_", None)
    if feat_names is not None:
        missing = [c for c in feat_names if c not in X_all.columns]
        if missing:
            raise KeyError(f"X_all is missing columns the model expects: {missing}")
        X_all = X_all[list(feat_names)]

    time_points = np.arange(0, 61, 1)
    n_subj = len(X_all)
    ncols  = min(n_subj, 4)
    nrows  = (n_subj + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    rows = []
    for subj_i, (subject_id, x_row) in enumerate(X_all.iterrows()):
        ax = axes[subj_i // ncols][subj_i % ncols]
        ax.set_title(f"Subject: {subject_id}", fontweight="bold", fontsize=12)

        for sc in TREATMENT_SCENARIOS:
            x_sc = x_row.copy()
            # Patch treatment features in the already-engineered feature space.
            # engineer_clinical() divides raw extent by EXTENT_NORMALIZER=3, so
            # scenario values 1/3, 2/3, 1.0 are already on the correct scale.
            x_sc["Extent of Tumor Resection"] = sc["resection"]
            x_sc["Chemotherapy"]              = sc["chemo"]

            surv_fn = estimator.predict_survival_function([x_sc.values])[0]
            curve   = np.array([_eval_surv(surv_fn, t) for t in time_points])
            pfs     = [_eval_surv(surv_fn, t) for t in SURV_TIME_POINTS]

            rows.append({
                "SubjectID": subject_id,
                "Scenario":  sc["label"],
                **{SURV_LABELS[i]: round(pfs[i], 3) for i in range(3)},
            })
            ax.plot(time_points, curve, label=sc["label"],
                    color=sc["color"], linestyle=sc["ls"], linewidth=2)

        ax.set_xlabel("Time (months)", fontsize=11)
        ax.set_ylabel("Progression-Free Survival", fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    for i in range(n_subj, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle("Treatment Scenario Simulations — DL-M1", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plot_path = output_dir / "treatment_scenarios.png"
    fig.savefig(str(plot_path), dpi=200, bbox_inches="tight")
    plt.close()

    df_scenarios = pd.DataFrame(rows)
    df_scenarios.to_csv(str(output_dir / "treatment_scenarios.csv"), index=False)
    print(df_scenarios.to_string(index=False))
    print(f"\n  Saved → {output_dir / 'treatment_scenarios.csv'}")
    print(f"  Saved → {plot_path}")
    return df_scenarios


# ─────────────────────────────────────────────────────────────────────────────
# Combine all results
# ─────────────────────────────────────────────────────────────────────────────

def combine_results(dlm1: pd.DataFrame, dlm2: pd.DataFrame | None,
                    scenarios: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = dlm1[["Risk Score", "Risk Group"]].rename(
        columns={"Risk Score": "DLM1_Risk_Score", "Risk Group": "DLM1_Risk_Group"}
    )
    if dlm2 is not None:
        summary = summary.join(dlm2[["DLM2_Risk_Score", "DLM2_Risk_Group"]])

    best = scenarios[scenarios["Scenario"].str.contains("Gross")].set_index("SubjectID")
    if not best.empty:
        summary = summary.join(
            best[SURV_LABELS].rename(columns={l: f"GTR_{l}" for l in SURV_LABELS}),
            how="left",
        )

    summary.to_csv(str(output_dir / "full_report.csv"))
    print(f"\n  Full report saved → {output_dir / 'full_report.csv'}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full pLGG Risk Inference: ResNet features → DL-M1 → DL-M2 → Treatment scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Molecular subtype choices (--mol_subtype):
  KIAA1549_BRAF  BRAF_V600E  NF1  FGFR  RTK  IDH  MYB  other_MAPK  wildtype
  unknown        (skips Step 3 / DL-M2)

Examples:
  python run_full_inference.py \\
      --image_dir     /data/T2w_scans \\
      --clinical_xlsx /data/Clinical_Features.xlsx \\
      --mol_subtype   KIAA1549_BRAF \\
      --output_dir    results/patient_report

  python run_full_inference.py \\
      --image_dir     /data/T2w_scans \\
      --clinical_xlsx /data/Clinical_Features.xlsx \\
      --mol_subtype   unknown \\
      --output_dir    results/patient_report
""",
    )
    parser.add_argument("--image_dir",     required=True,
                        help="Folder of T2w NIfTI files (.nii.gz or .nii)")
    parser.add_argument("--clinical_xlsx", required=True,
                        help="Clinical_Features.xlsx (see 01_DLM1_Clinico_ResNet/LGG_inference/ for format)")
    parser.add_argument("--mol_subtype",   required=True,
                        choices=list(MOL_SUBTYPE_MAP.keys()),
                        help="Molecular subtype for DL-M2; use 'unknown' to skip")
    parser.add_argument("--output_dir",    default="inference_report",
                        help="Output folder (default: inference_report/)")
    parser.add_argument("--model_path",    default=None,
                        help="Path to segmentation .pth (default: 05_Segmentation/final_model/final_model.pth)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("pLGG Full Risk Inference Pipeline")
    print("=" * 60)
    print(f"  Image dir        : {args.image_dir}")
    print(f"  Clinical xlsx    : {args.clinical_xlsx}")
    print(f"  Molecular subtype: {args.mol_subtype}")
    print(f"  Output dir       : {output_dir}")
    print(f"  Device           : {device}")

    step1_extract_features(args.image_dir, args.model_path, output_dir, device)

    dlm1_results, X_all, estimator, _ = step2_dlm1(args.clinical_xlsx, output_dir)

    dlm2_results = None
    if args.mol_subtype != "unknown":
        dlm2_results = step3_dlm2(dlm1_results, args.mol_subtype, output_dir)
    else:
        print("\nSTEP 3 — Skipped (mol_subtype = unknown)")

    scenarios = step4_treatment_scenarios(X_all, estimator, output_dir)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = combine_results(dlm1_results, dlm2_results, scenarios, output_dir)
    print(summary.to_string())

    print("\n" + "=" * 60)
    print(f"  All results saved to: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
