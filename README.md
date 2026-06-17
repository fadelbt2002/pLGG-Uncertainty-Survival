# Uncertainty-Aware Multimodal Survival Modeling in Pediatric Low-Grade Glioma

**Submitted to:** *American Journal of Neuroradiology (AJNR)*

---

## Overview

This repository contains the full code for an uncertainty-aware multimodal survival framework for risk stratification in **pediatric low-grade glioma (pLGG)**. The framework integrates deep learning imaging features extracted from a single T2-weighted MRI sequence with molecular subtype and clinical information, and evaluates whether molecular integration improves individualized risk stratification.

**Key contributions:**
- A regularized Cox survival model (DL-M1) combining ResNet imaging features + clinical data, achieving C-indices of 0.73 (discovery) and 0.70 (replication)
- A late-fusion molecular integration model (DL-M2) that improved replication-cohort discrimination and produced biologically coherent patient reclassification (BRAF V600E / KIAA1549-BRAF)
- Bootstrap resampling for per-patient prediction uncertainty quantification
- Treatment scenario simulations across four resection/chemotherapy combinations
- Statistical comparison demonstrating that a single-sequence deep learning pipeline performs comparably to a full multiparametric radiomic pipeline

---

## Data

Data were obtained from the [Children's Brain Tumor Network (CBTN)](https://cbtn.org/). Access to CBTN data requires a data access agreement through the [Pediatric Cancer Data Commons (PCDC)](https://commons.cri.uchicago.edu/pcdc/). Raw imaging and clinical data are **not included** in this repository.

---

## Repository Structure

```
├── run_full_inference.py           # ← END-TO-END: T2w NIfTI → DL-M1 → DL-M2 → treatment scenarios
│
├── 01_DLM1_Clinico_ResNet/         # DL-M1: imaging + clinical Cox model
│   ├── 1_Data_Wrangling.ipynb
│   ├── 2_Feature_Engineering.ipynb
│   ├── 3_Predictive_Modeling.ipynb
│   ├── 4_KM_Curves.ipynb
│   ├── inference.py                # Run risk prediction on new subjects
│   ├── encoder.pkl                 # Fitted preprocessing artifacts
│   ├── scaler_clinical.pkl
│   ├── scaler_radiomic.pkl
│   ├── imputer.pkl
│   ├── remover.pkl
│   ├── dropper.pkl
│   ├── LGG_inference/              # Example input spreadsheets for inference
│   └── outputs/                    # Trained model (estimator.pkl), metrics, plots
│
├── 02_Molecular_Subtype/           # Molecular-only survival model
│   ├── 1_Data_Wrangling.ipynb
│   ├── 2_Feature_Engineering.ipynb
│   ├── 3_Predictive_Modeling.ipynb
│   ├── molecular_subtype_encoder.pkl
│   ├── molecular_subtype_mapping.csv
│   └── outputs/
│
├── 03_Late_Fusion_DLM2/            # DL-M2: late fusion with molecular subtype
│   └── Late_Fusion_Modeling_ResNet.py
│
├── 04_Analysis/
│   ├── uncertainty_analysis_DLM1_vs_DLM2.py         # Bootstrap CI comparison M1 vs M2
│   ├── treatment_scenario_DLM1.py                    # Treatment simulations (DL-M1)
│   ├── treatment_scenario_DLM2.py                    # Treatment simulations (DL-M2)
│   ├── statistical_comparison_radiomic_vs_resnet.py  # Radiomic vs DL-M1 comparison
│   ├── risk_group_change.py                          # Patient reclassification M1→M2
│   └── permutation_sanity_check.py                   # Permutation test (reviewer validation)
│
└── 05_Segmentation/
    ├── train_segmentation.py                    # Fine-tune ResNet34 on pBT cohort (n=752)
    ├── test_segmentation.py                     # Evaluate segmentation model (Dice score)
    ├── extract_resnet_features_for_inference.py # ← START HERE: folder of NIfTI → ResNet_Features.xlsx
    ├── extract_features.py                      # Advanced: all layer/pooling variants
    ├── dataset.py              # CombinedSegmentationDataset (T2w, robust normalization)
    ├── model.py                # generate_model() — ResNet34 segmentation head
    ├── resnet_seg.py           # 3D ResNet backbone (resnet10 → resnet200)
    ├── utils.py                # Logging, Dice coefficient helpers
    └── final_model/            # Trained segmentation weights (Git LFS)
        ├── final_model.pth     # ResNet34 fine-tuned on n=752 pediatric brain tumors
        ├── config.json
        ├── summary.json
        └── evaluation/         # Test-set Dice scores and per-sample results
```

---

## Installation

```bash
git clone https://github.com/fadelbt2002/pLGG-Uncertainty-Survival.git
cd pLGG-Uncertainty-Survival
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## Running the Notebooks

Notebooks are numbered and should be run in order within each module:

1. `1_Data_Wrangling.ipynb` — load and harmonize clinical/imaging data
2. `2_Feature_Engineering.ipynb` — encode, scale, and select features
3. `3_Predictive_Modeling.ipynb` — train regularized Cox model with cross-validation
4. `4_KM_Curves.ipynb` — Kaplan-Meier stratification plots

---

## Full End-to-End Inference (Recommended)

Run the complete pipeline — from raw T2w MRI to risk report and treatment simulations — with a single command:

```bash
python run_full_inference.py \
  --image_dir     /path/to/T2w_nifti_files \
  --clinical_xlsx /path/to/Clinical_Features.xlsx \
  --mol_subtype   KIAA1549_BRAF \
  --output_dir    results/patient_report
```

This runs four steps automatically:
1. **ResNet feature extraction** — layer3 GAP+GMP from fine-tuned ResNet34 (512 features)
2. **DL-M1 risk score** — penalized Cox model (clinical + imaging)
3. **DL-M2 risk score** — late-fusion with molecular subtype model (omit with `--mol_subtype unknown`)
4. **Treatment scenario simulations** — 4 resection × chemotherapy combinations → survival curves

Outputs are written to `--output_dir`: `ResNet_Features.xlsx`, `DLM1_risk_results.csv`, `DLM2_risk_results.csv`, `treatment_scenarios.png`, `treatment_scenarios.csv`, `full_report.csv`.

Molecular subtype choices: `KIAA1549_BRAF`, `BRAF_V600E`, `NF1`, `FGFR`, `RTK`, `IDH`, `MYB`, `other_MAPK`, `wildtype`, `unknown`.

See `01_DLM1_Clinico_ResNet/LGG_inference/` for the expected `Clinical_Features.xlsx` column format.

---

## Step-by-Step Inference (Alternative)

To run individual steps manually:

```bash
cd 01_DLM1_Clinico_ResNet
python inference.py \
  --input-dir   LGG_inference \
  --artifact-dir . \
  --model-dir   outputs/models \
  --output-dir  LGG_inference
```

Input files required (see `LGG_inference/` for format):
- `Clinical_Features.xlsx` — subject clinical variables
- `ResNet_Features.xlsx` — ResNet features extracted from T2w MRI (see `05_Segmentation/`)

---

## Segmentation and Feature Extraction

The segmentation model (ResNet34) was fine-tuned on a pediatric brain tumor cohort (n=752) using T2w MRI only. Trained weights are provided in `05_Segmentation/final_model/final_model.pth` (stored via Git LFS).

**Step 1 — Extract ResNet features from your T2w MRI folder:**

```bash
cd 05_Segmentation
python extract_resnet_features_for_inference.py \
  --image_dir /path/to/T2w_nifti_files \
  --output_dir /path/to/results
```

- Input: a folder of T2w NIfTI files (`.nii.gz` or `.nii`) — no manifest required.
  SubjectIDs are parsed automatically from filenames (e.g. `C1234567_T2.nii.gz` → `C1234567`).
- Output: `ResNet_Features.xlsx` with `SubjectID` + 512 features (layer3 GAP+GMP) — the exact format expected by `inference.py`.

> **Note:** `final_model/final_model.pth` is stored via Git LFS. If the file is missing after cloning, run `git lfs pull`.

**Step 2 — Run DL-M1 risk inference:**

```bash
cd ../01_DLM1_Clinico_ResNet
python inference.py \
  --input-dir  /path/to/results \
  --output-dir /path/to/results
```

Also place `Clinical_Features.xlsx` in the same `--input-dir` (see `LGG_inference/` for the expected column layout).

**Advanced: full feature extraction** (all layer/pooling variants for retraining):

```bash
cd 05_Segmentation
python extract_features.py \
  --excel_path /path/to/subjects.xlsx \
  --image_dirs /path/to/T2w_images \
  --output_dir /path/to/features
```

**To retrain the segmentation model** on your own cohort:

```bash
python train_segmentation.py \
  --data_csv   /path/to/dataset.csv \
  --output_dir /path/to/run_output \
  --model_depth 34
```

---

## Citation

> Batal F, et al. *Uncertainty-Aware Multimodal Survival Modeling in Pediatric Low-Grade Glioma.* AJNR, 2026. (under review)

---

## License

This code is released under the MIT License. Model weights and preprocessing artifacts are provided for research use only.
