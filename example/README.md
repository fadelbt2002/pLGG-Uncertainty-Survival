# Example Inputs

This folder contains synthetic example files showing the exact format required by `run_full_inference.py`.

---

## Files

### `Clinical_Features.xlsx`

Three synthetic subjects demonstrating the 8 required columns and the range of valid values:

| Column | SUBJ001 | SUBJ002 | SUBJ003 |
|--------|---------|---------|---------|
| SubjectID | SUBJ001 | SUBJ002 | SUBJ003 |
| legal_sex | Male | Female | Male |
| age_at_event_days | 2920 (~8 yr) | 4380 (~12 yr) | 1825 (~5 yr) |
| consolidated_tumor_locations | Cerebellar | Suprasellar | Brain Stem |
| cancer_predisposition | None documented | None documented | Neurofibromatosis, Type 1 (NF-1) |
| extent_of_tumor_resection | Gross/Near total resection | Biopsy only | Partial resection |
| chemotherapy | No | Yes | Yes |
| radiation | No | No | No |

Replace these rows with your own subjects. Each row = one patient.

**SubjectID must match the prefix of the NIfTI filename.** For example, if your scan is named `SUBJ001_T2_norm.nii.gz`, the SubjectID must be `SUBJ001`.

---

### `ResNet_Features_template.xlsx`

A placeholder showing the expected column layout (`SubjectID` + `feature_000` … `feature_511`). The values are all zeros — **do not use this file for real inference**.

Real ResNet features are generated automatically in Step 1:

```bash
python run_full_inference.py \
  --image_dir     /path/to/T2w_nifti_files \
  --clinical_xlsx example/Clinical_Features.xlsx \
  --mol_subtype   KIAA1549_BRAF \
  --output_dir    results/
```

Step 1 writes `ResNet_Features.xlsx` to `--output_dir` automatically. Use `--skip_step1` on subsequent runs to reuse it.

---

## Quick-start with your own data

1. Copy `Clinical_Features.xlsx` and fill in your subject(s).
2. Place skull-stripped T2w NIfTI files in a folder (one file per subject).
3. Run:

```bash
python run_full_inference.py \
  --image_dir     /path/to/T2w_scans \
  --clinical_xlsx example/Clinical_Features.xlsx \
  --mol_subtype   KIAA1549_BRAF \
  --output_dir    results/my_report
```

See the top-level [README](../README.md) for the full list of valid values for each column and all `--mol_subtype` options.
