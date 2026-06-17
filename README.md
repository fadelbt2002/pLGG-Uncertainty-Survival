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
│   └── outputs/                    # Trained model, metrics, plots
│
├── 02_Molecular_Subtype/           # Molecular-only survival model
│   ├── 1_Data_Wrangling.ipynb
│   ├── 2_Feature_Engineering.ipynb
│   ├── 3_Predictive_Modeling.ipynb
│   └── outputs/
│
├── 03_Late_Fusion_DLM2/            # DL-M2: late fusion with molecular subtype
│   └── Late_Fusion_Modeling_ResNet.py
│
├── 04_Analysis/
│   ├── uncertainty_analysis_DLM1_vs_DLM2.py    # Bootstrap CI comparison M1 vs M2
│   ├── treatment_scenario_DLM1.py               # Treatment simulations (DL-M1)
│   ├── treatment_scenario_DLM2.py               # Treatment simulations (DL-M2)
│   ├── statistical_comparison_radiomic_vs_resnet.py  # Radiomic vs DL-M1 comparison
│   ├── risk_group_change.py                     # Patient reclassification M1→M2
│   └── permutation_sanity_check.py              # Permutation test (reviewer validation)
│
└── 05_Segmentation/
    ├── train_segmentation.py       # Fine-tuning tumor segmentation on pBT cohort
    └── test_segmentation.py        # Inference + feature extraction from T2w MRI
```

---

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
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

## Inference on New Subjects

To apply the trained DL-M1 model to new patients:

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

The segmentation model was fine-tuned on a pediatric brain tumor cohort (n=752) using nnU-Net. ResNet features are extracted from the segmented tumor region on T2w MRI only.

See `05_Segmentation/` for training and inference scripts.

---

## Citation

> Batal F, et al. *Uncertainty-Aware Multimodal Survival Modeling in Pediatric Low-Grade Glioma.* AJNR, 2026. (under review)

---

## License

This code is released under the MIT License. Model weights and preprocessing artifacts are provided for research use only.
