# Uncertainty-Aware Multimodal Survival Modeling in Pediatric Low-Grade Glioma

**Submitted to:** *American Journal of Neuroradiology (AJNR)*

---

## Overview

This repository contains the full code for an uncertainty-aware multimodal survival framework for risk stratification in **pediatric low-grade glioma (pLGG)**. The framework integrates deep learning imaging features extracted from a single T2-weighted MRI sequence with molecular subtype and clinical information, and evaluates whether molecular integration improves individualized risk stratification.

**Key contributions:**
- A regularized Cox survival model (DL-M1) combining ResNet imaging features + clinical data, achieving C-indices of 0.73 (discovery) and 0.70 (replication)
- A late-fusion molecular integration model (DL-M2) that improved replication-cohort discrimination and produced biologically coherent patient reclassification (BRAF V600E / KIAA1549-BRAF)
- Bootstrap resampling for per-patient prediction uncertainty quantification (95% CI from 1,000 resamples)
- Treatment scenario simulations across four resection/chemotherapy combinations with CI bands
- Statistical comparison demonstrating that a single-sequence deep learning pipeline performs comparably to a full multiparametric radiomic pipeline

---

## Data

Data were obtained from the [Children's Brain Tumor Network (CBTN)](https://cbtn.org/). Access to CBTN data requires a data access agreement through the [Pediatric Cancer Data Commons (PCDC)](https://commons.cri.uchicago.edu/pcdc/). Raw imaging and clinical data are **not included** in this repository.

---

## Repository Structure

```
├── run_full_inference.py           # END-TO-END: T2w NIfTI → DL-M1 → DL-M2 → treatment scenarios + 95% CI
│
├── example/                        # Synthetic example inputs — start here
│   ├── Clinical_Features.xlsx      # 3 example subjects showing all valid column values
│   └── ResNet_Features_template.xlsx  # Column layout only (zeros); real features from Step 1
│
├── 01_DLM1_Clinico_ResNet/         # DL-M1: imaging + clinical Cox model
│   ├── 1_Data_Wrangling.ipynb
│   ├── 2_Feature_Engineering.ipynb
│   ├── 3_Predictive_Modeling.ipynb
│   ├── 4_KM_Curves.ipynb
│   ├── inference.py                # Run DL-M1 risk prediction on new subjects
│   ├── encoder.pkl                 # Fitted preprocessing artifacts
│   ├── scaler_clinical.pkl
│   ├── scaler_radiomic.pkl
│   ├── imputer.pkl
│   ├── remover.pkl
│   ├── dropper.pkl
│   ├── LGG_inference/              # Example Clinical_Features.xlsx and ResNet_Features.xlsx
│   └── outputs/models/             # estimator.pkl, bootstrap_models.pkl (1000 × CoxPH), risk_threshold.pkl,
│                                   # refit_coxph_model.pkl (CoxPH α=0, 12 features), refit_threshold.pkl
│
├── 02_Molecular_Subtype/           # Molecular + clinical survival model
│   ├── 1_Data_Wrangling.ipynb
│   ├── 2_Feature_Engineering.ipynb
│   ├── 3_Predictive_Modeling.ipynb
│   ├── molecular_subtype_encoder.pkl  # multi-hot encoding spec (TOKEN_MAP + column order)
│   ├── molecular_subtype_mapping.csv
│   ├── scaler_clinical.pkl            # StandardScaler for Age at Diagnosis (molecular cohort)
│   └── outputs/models/             # estimator.pkl (CoxnetSurvivalAnalysis, 14 features)
│
├── 03_Late_Fusion_DLM2/            # DL-M2: late fusion with molecular subtype
│   ├── Late_Fusion_Modeling_ResNet.py
│   └── final_model/
│       ├── model.pkl               # DL-M2 CoxPH fusion model
│       ├── scalers.pkl             # Discovery-fit StandardScalers (CR and Molecular)
│       ├── threshold.txt           # Median risk score threshold (discovery cohort)
│       ├── fusion_weights.csv      # CR vs Molecular contribution weights
│       ├── performance_summary.csv # C-indices for base and fusion models
│       └── bootstrap_entries.pkl   # 1000 bootstrap entries (model + scaler_cr + scaler_mol)
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
    ├── extract_resnet_features_for_inference.py # Folder of T2w NIfTI → ResNet_Features.xlsx
    ├── extract_features.py                      # Advanced: all layer/pooling variants
    ├── dataset.py              # CombinedSegmentationDataset (T2w, robust normalization)
    ├── model.py                # generate_model() — ResNet34 segmentation head
    ├── resnet_seg.py           # 3D ResNet backbone (resnet10 → resnet200)
    ├── utils.py                # Logging, Dice coefficient helpers
    └── final_model/            # Trained segmentation weights (Git LFS)
        ├── final_model.pth     # ResNet34 fine-tuned on n=752 pediatric brain tumors
        ├── config.json
        ├── summary.json
        └── evaluation/
```

---

## Installation

```bash
git clone https://github.com/fadelbt2002/pLGG-Uncertainty-Survival.git
cd pLGG-Uncertainty-Survival
git lfs pull          # download final_model.pth (~242 MB, requires Git LFS)
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## Preprocessing: Skull-Stripping and Normalization (Step 0)

Before running inference, T2w MRI volumes must be **skull-stripped and intensity-normalized**. This is the same preprocessing applied to all training data. We recommend the D3B pediatric brain auto-segmentation pipeline:

**[d3b-center/peds-brain-auto-seg-public](https://github.com/d3b-center/peds-brain-auto-seg-public)**

This pipeline performs skull stripping and produces normalized NIfTI outputs ready for feature extraction. Follow the instructions in that repository to preprocess your T2w volumes before proceeding.

> The segmentation model (`final_model.pth`) was trained on outputs from this preprocessing pipeline. Using a different skull-stripping method may affect feature quality.

---

## Preparing Inputs for Inference

The pipeline requires three inputs. Two are straightforward; the clinical spreadsheet needs a bit of preparation.

### Input 1 — T2w MRI (NIfTI)

Place skull-stripped, normalized T2w volumes in a folder. Any of these naming conventions work:

```
C1234567_T2_ss_norm.nii.gz   → SubjectID = C1234567
sub-001_T2.nii.gz            → SubjectID = sub-001
C9876543.nii.gz              → SubjectID = C9876543
```

SubjectIDs are parsed automatically from filenames — no manifest is needed.

---

### Input 2 — ResNet Features (auto-generated, output-ready)

ResNet features are extracted automatically by the pipeline (Step 1). You do not need to prepare this file manually — `run_full_inference.py` generates `ResNet_Features.xlsx` directly from your T2w folder and saves it to `--output_dir`.

If you have already extracted features in a previous run and want to skip Step 1:

```bash
python run_full_inference.py ... --skip_step1
```

The file must be in `--output_dir` as `ResNet_Features.xlsx` with columns `SubjectID`, `feature_000` … `feature_511` (512 columns = layer3 GAP + GMP from ResNet34).

---

### Input 3 — Clinical Features Spreadsheet

Create an Excel file (`Clinical_Features.xlsx`) with **exactly these 8 columns** — no extras needed:

| Column | Type | Description |
|--------|------|-------------|
| `SubjectID` | string | Must match the prefix parsed from the NIfTI filename (e.g. `C1234567`) |
| `legal_sex` | string | `Female` or `Male` |
| `age_at_event_days` | integer | Age at imaging in **days** (e.g. 3650 = ~10 years) |
| `consolidated_tumor_locations` | string | Tumor location — see valid values below |
| `cancer_predisposition` | string | NF1 status — see valid values below |
| `extent_of_tumor_resection` | string | Surgical extent — see valid values below |
| `chemotherapy` | string | Chemotherapy received — see valid values below |
| `radiation` | string | Radiation received — see valid values below |
| `molecular_subtype` | string | Molecular driver — see valid values below. Leave blank or write `unknown` to skip DL-M2 for that subject. |

**Valid values for each categorical column:**

**`consolidated_tumor_locations`** — must exactly match one of:
```
Cerebellar        Lobar            Suprasellar     Brain Stem
Multi regional    Intra Ventricular  Thalamus      Basal Ganglia
Other
```
> Values are count-encoded using the discovery cohort frequency. An unseen value will produce a warning and a NaN risk score for that subject.

**`cancer_predisposition`** — must exactly match one of:
```
None documented
Neurofibromatosis, Type 1 (NF-1)
```
> All other strings are treated as "None documented" (NF1 = 0).

**`extent_of_tumor_resection`** — must exactly match one of:
```
Biopsy only
Partial resection
Gross/Near total resection
Not Applicable
Unavailable
```

**`chemotherapy`** and **`radiation`** — `Yes` or `No`

**Example `Clinical_Features.xlsx`** (see `01_DLM1_Clinico_ResNet/LGG_inference/` for a ready-to-use template):

| SubjectID | legal_sex | age_at_event_days | consolidated_tumor_locations | cancer_predisposition | extent_of_tumor_resection | chemotherapy | radiation |
|-----------|-----------|------------------|------------------------------|-----------------------|--------------------------|--------------|-----------|
| C1277724 | Male | 4762 | Cerebellar | None documented | Gross/Near total resection | No | No |

Multiple subjects can be listed as additional rows — the pipeline scores all of them in one run.

---

### Input 4 — Molecular Subtype (spreadsheet column, recommended)

Add a `molecular_subtype` column to `Clinical_Features.xlsx`. Each subject can have a different subtype, which is the recommended approach for multi-subject runs.

Valid values:

| Argument value | Molecular alteration |
|----------------|----------------------|
| `KIAA1549_BRAF` | KIAA1549::BRAF fusion (most common pLGG driver) |
| `BRAF_V600E` | BRAF V600E point mutation |
| `NF1` | NF1 loss |
| `FGFR` | FGFR alteration |
| `RTK` | Other receptor tyrosine kinase |
| `IDH` | IDH mutation |
| `MYB` | MYB/MYBL1 alteration |
| `other_MAPK` | Other MAPK pathway |
| `wildtype` | No known driver (wildtype / other) |
| `CDKN2A_B` | CDKN2A/B co-deletion |
| `unknown` | Molecular status unavailable — skips DL-M2 and runs DL-M1 only |

**Co-driver tumors:** a subject carrying more than one alteration can list them
comma-separated in the same cell — for example `KIAA1549_BRAF, CDKN2A_B`. Each
recognised token fires its own binary feature (multi-hot encoding), matching how
the model was trained. `wildtype` fires none and acts as the all-zeros reference.

The `--mol_subtype` CLI flag is a fallback that applies the same subtype to all subjects when the column is absent from the spreadsheet. If molecular testing has not been performed, leave the `molecular_subtype` cell blank or write `unknown` — the pipeline will run Steps 1, 2, and 4a (DL-M1 only) for that subject.

---

## Full End-to-End Inference

```bash
python run_full_inference.py \
  --image_dir     /path/to/T2w_nifti_files \
  --clinical_xlsx /path/to/Clinical_Features.xlsx \
  --mol_subtype   KIAA1549_BRAF \
  --output_dir    results/patient_report
```

**What runs** (after Step 0 preprocessing):

| Step | Description | Output |
|------|-------------|--------|
| 1 | ResNet34 feature extraction (layer3 GAP+GMP, 512-dim) | `ResNet_Features.xlsx` |
| 2 | DL-M1 risk score + 95% CI (1,000 bootstrap CoxPH resamples) | `DLM1_risk_results.csv` |
| 3 | DL-M2 fusion risk score + 95% CI (1,000 bootstrap fusion entries) | `DLM2_risk_results.csv` |
| 4a | DL-M1 treatment scenarios + CI bands (4 resection × chemo combinations) | `DLM1_treatment_scenarios.csv`, `<ID>_DLM1_scenarios.png` |
| 4b | DL-M2 treatment scenarios + CI bands | `DLM2_treatment_scenarios.csv`, `<ID>_DLM2_scenarios.png` |
| — | Combined summary table | `full_report.csv` |

**Skip feature extraction** (if `ResNet_Features.xlsx` already exists in `--output_dir`):

```bash
python run_full_inference.py \
  --image_dir     /path/to/T2w_nifti_files \
  --clinical_xlsx /path/to/Clinical_Features.xlsx \
  --mol_subtype   KIAA1549_BRAF \
  --output_dir    results/patient_report \
  --skip_step1
```

**Without molecular subtype** (runs Steps 1, 2, 4a only):

```bash
python run_full_inference.py \
  --image_dir     /path/to/T2w_nifti_files \
  --clinical_xlsx /path/to/Clinical_Features.xlsx \
  --mol_subtype   unknown \
  --output_dir    results/patient_report
```

---

## Bootstrap Uncertainty Methodology

95% confidence intervals follow the same bootstrap-the-training-set procedure used for the replication cohort in the paper:

- **DL-M1 CI**: 1,000 CoxPH models (α=0), each fit on a bootstrap resample of the discovery cohort using the LASSO-selected feature subset (12 features). The point estimate comes from `refit_coxph_model.pkl` — an unregularized CoxPH (α=0) refit on all n=282 discovery patients using the same 12 features — ensuring the point estimate is guaranteed to fall within its own CI band. Stored in `01_DLM1_Clinico_ResNet/outputs/models/bootstrap_models.pkl`, `refit_coxph_model.pkl`, and `refit_threshold.pkl`.

- **DL-M2 CI**: 1,000 bootstrap fusion entries, each containing a CoxPH model and its own per-resample StandardScalers for the CR and Molecular risk scores. Each entry scales the new subject's raw risk scores independently before predicting, ensuring consistency within each bootstrap replicate. Stored in `03_Late_Fusion_DLM2/final_model/bootstrap_entries.pkl`.

CIs reflect **parameter/sampling uncertainty** from finite training-sample size, not individual outcome variability.

---

## Step-by-Step Inference (Alternative)

To run DL-M1 inference standalone without the full pipeline:

```bash
# Step 1 — extract ResNet features
cd 05_Segmentation
python extract_resnet_features_for_inference.py \
  --image_dir  /path/to/T2w_nifti_files \
  --output_dir /path/to/results

# Step 2 — run DL-M1 risk prediction
cd ../01_DLM1_Clinico_ResNet
python inference.py \
  --input-dir  /path/to/results \
  --output-dir /path/to/results
```

Place `Clinical_Features.xlsx` in the same `--input-dir`. See `LGG_inference/` for a ready-to-use template.

---

## Running the Notebooks

Notebooks are numbered and should be run in order within each module:

1. `1_Data_Wrangling.ipynb` — load and harmonize clinical/imaging data
2. `2_Feature_Engineering.ipynb` — encode, scale, and select features
3. `3_Predictive_Modeling.ipynb` — train regularized Cox model with cross-validation
4. `4_KM_Curves.ipynb` — Kaplan-Meier stratification plots

---

## Citation

> Batal F, et al. *Uncertainty-Aware Multimodal Survival Modeling in Pediatric Low-Grade Glioma.* AJNR, 2026. (under review)

---

## License

This code is released under the MIT License. Model weights and preprocessing artifacts are provided for research use only.
