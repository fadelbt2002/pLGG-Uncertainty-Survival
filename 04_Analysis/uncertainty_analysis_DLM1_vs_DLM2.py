"""
Bootstrap Uncertainty Analysis for pLGG Survival Models
Compares DL-M1 (Clinical-ResNet) vs DL-M2 (Clinical-ResNet + Molecular) models

M1: Bootstrap on the FULL ResNet discovery patients using plain CoxPH on the
    LASSO-selected features (fixed, not re-selected per iteration).
    All predictions and metrics evaluated on the intersection cohort only.

M2: Bootstrap on the intersection discovery patients using plain CoxPH
    on 2 standardized risk scores (ClinResNet + Molecular).
    Scalers are refit on each bootstrap sample to maintain consistency between
    normalization and training data within each iteration.
    All predictions and metrics evaluated on the same intersection patients.

Survival probability uncertainty evaluated at 60 months (5-year PFS),
the standard clinical endpoint for pLGG.

Author: Fadel Batal
Date: 02/11/2026
"""

import pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sklearn.preprocessing import StandardScaler
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
# Repo root: one level up from this script's folder (04_Analysis/)
REPO_ROOT   = Path(__file__).resolve().parent.parent
DLM1_DIR    = REPO_ROOT / '01_DLM1_Clinico_ResNet'
MOL_DIR     = REPO_ROOT / '02_Molecular_Subtype'

N_BOOTSTRAP = 1000
TIME_POINTS  = np.array([12, 24, 36, 48, 60])
OUTPUT_DIR   = str(REPO_ROOT / '04_Analysis' / 'outputs' / 'uncertainty_DLM1_vs_DLM2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ── Helper Functions ───────────────────────────────────────────────────────────

def extract_survival_at_times(surv_funcs, time_points):
    return np.array([[fn(t) for t in time_points] for fn in surv_funcs])


def convert_to_structured_array(y_df):
    return np.array(
        [(bool(e), float(t)) for e, t in
         zip(y_df['Event'], y_df['Progression Free Survival'])],
        dtype=[('Event', bool), ('Progression Free Survival', float)]
    )


def compute_summary_stats(df, model_name):
    results = []
    for cohort in ['Discovery', 'Replicate']:
        cohort_df = df[df['Cohort'] == cohort]
        results.append({
            'Model': model_name,
            'Cohort': cohort,
            'N_Patients': len(cohort_df),
            'Risk_CI_Width_Median':   cohort_df['Risk_CI_Width'].median(),
            'Risk_CI_Width_Mean':     cohort_df['Risk_CI_Width'].mean(),
            'Risk_CI_Width_IQR':      cohort_df['Risk_CI_Width'].quantile(0.75) - cohort_df['Risk_CI_Width'].quantile(0.25),
            'Risk_CI_Width_Q25':      cohort_df['Risk_CI_Width'].quantile(0.25),
            'Risk_CI_Width_Q75':      cohort_df['Risk_CI_Width'].quantile(0.75),
            'Risk_CI_Width_Min':      cohort_df['Risk_CI_Width'].min(),
            'Risk_CI_Width_Max':      cohort_df['Risk_CI_Width'].max(),
            'Surv60_CI_Width_Median': cohort_df['Surv_60m_CI_Width'].median(),
            'Surv60_CI_Width_Mean':   cohort_df['Surv_60m_CI_Width'].mean(),
            'Surv60_CI_Width_IQR':    cohort_df['Surv_60m_CI_Width'].quantile(0.75) - cohort_df['Surv_60m_CI_Width'].quantile(0.25),
            'Surv60_CI_Width_Q25':    cohort_df['Surv_60m_CI_Width'].quantile(0.25),
            'Surv60_CI_Width_Q75':    cohort_df['Surv_60m_CI_Width'].quantile(0.75),
        })
    return pd.DataFrame(results)


def compute_c_index_summary(df, model_name):
    results = []
    for cohort_type in ['Discovery', 'Replicate']:
        harrell_valid = df[f'Harrell_{cohort_type}'].dropna()
        uno_valid     = df[f'Uno_{cohort_type}'].dropna()
        results.append({
            'Model':             model_name,
            'Cohort':            cohort_type,
            'N_Bootstrap':       len(df),
            'N_Successful':      len(harrell_valid),
            'N_Uno_Successful':  len(uno_valid),
            'Harrell_Mean':      harrell_valid.mean(),
            'Harrell_Median':    harrell_valid.median(),
            'Harrell_Std':       harrell_valid.std(),
            'Harrell_CI_2.5th':  harrell_valid.quantile(0.025),
            'Harrell_CI_97.5th': harrell_valid.quantile(0.975),
            'Harrell_CI_Width':  harrell_valid.quantile(0.975) - harrell_valid.quantile(0.025),
            'Uno_Mean':          uno_valid.mean(),
            'Uno_Median':        uno_valid.median(),
            'Uno_Std':           uno_valid.std(),
            'Uno_CI_2.5th':      uno_valid.quantile(0.025),
            'Uno_CI_97.5th':     uno_valid.quantile(0.975),
            'Uno_CI_Width':      uno_valid.quantile(0.975) - uno_valid.quantile(0.025),
        })
    return pd.DataFrame(results)


def build_patient_uncertainty(boot_pred, boot_surv, ids, Y_df, cohort_label, model_label):
    rows = []
    for i, pid in enumerate(ids):
        preds = boot_pred[:, i]
        preds = preds[~np.isnan(preds)]
        row = {
            'SubjectID':      pid,
            'Cohort':         cohort_label,
            'Model':          model_label,
            'Risk_Mean':      np.mean(preds),
            'Risk_Median':    np.median(preds),
            'Risk_Std':       np.std(preds),
            'Risk_CI_2.5th':  np.percentile(preds, 2.5),
            'Risk_CI_97.5th': np.percentile(preds, 97.5),
            'Risk_CI_Width':  np.percentile(preds, 97.5) - np.percentile(preds, 2.5),
            'PFS':            Y_df.iloc[i]['Progression Free Survival'],
            'Event':          Y_df.iloc[i]['Event'],
        }
        for t_idx, t in enumerate(TIME_POINTS):
            s = boot_surv[:, i, t_idx]
            s = s[~np.isnan(s)]
            row[f'Surv_{t}m_Mean']      = np.mean(s)
            row[f'Surv_{t}m_CI_2.5th']  = np.percentile(s, 2.5)
            row[f'Surv_{t}m_CI_97.5th'] = np.percentile(s, 97.5)
            row[f'Surv_{t}m_CI_Width']  = np.percentile(s, 97.5) - np.percentile(s, 2.5)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Load Data ──────────────────────────────────────────────────────────────────

print("=" * 80)
print("BOOTSTRAP UNCERTAINTY ANALYSIS: M1 vs M2 (Clinico-ResNet)")
print("=" * 80)
print("\nLoading data...")

with open(DLM1_DIR / 'outputs' / 'models' / 'X_clinical.pkl', 'rb') as f:
    X_clinical = pickle.load(f)
with open(DLM1_DIR / 'outputs' / 'models' / 'X_resnet.pkl', 'rb') as f:
    X_resnet = pickle.load(f)
with open(DLM1_DIR / 'outputs' / 'models' / 'y.pkl', 'rb') as f:
    Y_resnet = pickle.load(f)
with open(DLM1_DIR / 'outputs' / 'models' / 'estimator.pkl', 'rb') as f:
    trained_model_resnet = pickle.load(f)

X_clinical_features   = X_clinical.drop(columns=['Cohort'], errors='ignore')
X_resnet_features     = X_resnet.drop(columns=['Cohort'], errors='ignore')
X_clinresnet_combined = pd.concat([X_clinical_features, X_resnet_features], axis=1).sort_index()
Y_resnet              = Y_resnet.sort_index()

print(f"Clinical-ResNet loaded: {X_clinresnet_combined.shape}")
print(f"  Full Discovery: {len(Y_resnet[Y_resnet['Cohort']=='Discovery'])}")
print(f"  Full Replicate: {len(Y_resnet[Y_resnet['Cohort']=='Replicate'])}")

# ── LASSO-selected features ────────────────────────────────────────────────────
original_coefs_resnet    = trained_model_resnet.coef_[:, 0]
selected_feature_mask    = original_coefs_resnet != 0
selected_feature_indices = np.where(selected_feature_mask)[0]
selected_feature_names   = X_clinresnet_combined.columns[selected_feature_indices].tolist()
n_selected               = len(selected_feature_indices)

print(f"LASSO-selected features: {n_selected} / {len(original_coefs_resnet)}")
print("  Fixed for all M1 bootstrap iterations — no re-selection")

# ── Molecular data ─────────────────────────────────────────────────────────────
for sub in ['outputs/models', '']:
    try:
        with open(MOL_DIR / sub / 'X_molecular.pkl', 'rb') as f:
            X_molecular = pickle.load(f)
        with open(MOL_DIR / sub / 'y.pkl', 'rb') as f:
            Y_molecular = pickle.load(f)
        break
    except FileNotFoundError:
        continue

X_molecular = X_molecular.sort_index()
Y_molecular = Y_molecular.sort_index()

print(f"Molecular data: {X_molecular.shape}")
print(f"  Discovery: {len(Y_molecular[Y_molecular['Cohort']=='Discovery'])}")
print(f"  Replicate: {len(Y_molecular[Y_molecular['Cohort']=='Replicate'])}")


# ── Patient Sets ───────────────────────────────────────────────────────────────

print("\nBuilding patient sets...")

full_discovery_ids = sorted(Y_resnet[Y_resnet['Cohort'] == 'Discovery'].index)
n_full_disc        = len(full_discovery_ids)

common_all_ids       = sorted(set(X_clinresnet_combined.index) & set(X_molecular.index))
common_discovery_ids = [p for p in common_all_ids if Y_resnet.loc[p, 'Cohort'] == 'Discovery']
common_replicate_ids = [p for p in common_all_ids if Y_resnet.loc[p, 'Cohort'] == 'Replicate']
n_inter_disc         = len(common_discovery_ids)
n_inter_rep          = len(common_replicate_ids)

print(f"  M1 bootstrap pool : Discovery = {n_full_disc}")
print(f"  M2 bootstrap pool : Discovery = {n_inter_disc}")
print(f"  Evaluation (both) : Discovery = {n_inter_disc}  Replicate = {n_inter_rep}")


# ── Build Structured Arrays ────────────────────────────────────────────────────

Y_full_disc     = Y_resnet.loc[full_discovery_ids]
X_full_disc     = X_clinresnet_combined.loc[full_discovery_ids, selected_feature_names]
y_full_disc_str = convert_to_structured_array(Y_full_disc)

Y_inter_disc     = Y_resnet.loc[common_discovery_ids]
Y_inter_rep      = Y_resnet.loc[common_replicate_ids]
X_inter_disc_M1  = X_clinresnet_combined.loc[common_discovery_ids, selected_feature_names]
X_inter_rep_M1   = X_clinresnet_combined.loc[common_replicate_ids,  selected_feature_names]
y_inter_disc_str = convert_to_structured_array(Y_inter_disc)
y_inter_rep_str  = convert_to_structured_array(Y_inter_rep)

X_full_disc_arr     = X_full_disc.values
X_inter_disc_M1_arr = X_inter_disc_M1.values
X_inter_rep_M1_arr  = X_inter_rep_M1.values

# ── Fixed tau for Uno C-index (computed once, used in both M1 and M2) ──────────
# tau = max observed event time in the intersection discovery set.
# Fixing this across all bootstrap iterations ensures a consistent truncation
# horizon and prevents failures when bootstrap max-time < evaluation max-time.
TAU = float(y_inter_disc_str['Progression Free Survival'][
            y_inter_disc_str['Event']].max())
print(f"\nUno C-index truncation horizon (TAU): {TAU:.1f} months")

# ── Pre-generate bootstrap indices (separated RNG streams for reproducibility) ─
rng_m1 = np.random.RandomState(42)
rng_m2 = np.random.RandomState(123)

M1_boot_indices = [
    rng_m1.choice(n_full_disc,  size=n_full_disc,  replace=True)
    for _ in range(N_BOOTSTRAP)
]
M2_boot_indices = [
    rng_m2.choice(n_inter_disc, size=n_inter_disc, replace=True)
    for _ in range(N_BOOTSTRAP)
]


# ══════════════════════════════════════════════════════════════════════════════
# M1 BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("M1: BOOTSTRAP UNCERTAINTY ANALYSIS (Clinico-ResNet)")
print(f"  Bootstrap pool : {n_full_disc} full discovery patients")
print(f"  Evaluation on  : {n_inter_disc} intersection discovery + {n_inter_rep} replicate")
print("=" * 80)

bs_pred_disc = np.zeros((N_BOOTSTRAP, n_inter_disc))
bs_pred_rep  = np.zeros((N_BOOTSTRAP, n_inter_rep))
bs_surv_disc = np.zeros((N_BOOTSTRAP, n_inter_disc, len(TIME_POINTS)))
bs_surv_rep  = np.zeros((N_BOOTSTRAP, n_inter_rep,  len(TIME_POINTS)))
bs_coefs     = np.zeros((N_BOOTSTRAP, n_selected))
bs_har_disc  = np.full(N_BOOTSTRAP, np.nan)
bs_har_rep   = np.full(N_BOOTSTRAP, np.nan)
bs_uno_disc  = np.full(N_BOOTSTRAP, np.nan)
bs_uno_rep   = np.full(N_BOOTSTRAP, np.nan)

print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations...")

for i in tqdm(range(N_BOOTSTRAP), desc="M1 Bootstrap"):
    boot_idx = M1_boot_indices[i]
    X_boot   = X_full_disc_arr[boot_idx]
    y_boot   = y_full_disc_str[boot_idx]

    try:
        m = CoxPHSurvivalAnalysis(alpha=0.0, verbose=0)
        m.fit(X_boot, y_boot)

        bs_coefs[i, :] = m.coef_

        pred_disc = m.predict(X_inter_disc_M1_arr)
        pred_rep  = m.predict(X_inter_rep_M1_arr)
        bs_pred_disc[i, :] = pred_disc
        bs_pred_rep[i, :]  = pred_rep

        bs_surv_disc[i, :, :] = extract_survival_at_times(
            m.predict_survival_function(X_inter_disc_M1_arr), TIME_POINTS)
        bs_surv_rep[i, :, :]  = extract_survival_at_times(
            m.predict_survival_function(X_inter_rep_M1_arr),  TIME_POINTS)

        bs_har_disc[i] = concordance_index_censored(
            y_inter_disc_str['Event'],
            y_inter_disc_str['Progression Free Survival'],
            pred_disc
        )[0]
        bs_har_rep[i] = concordance_index_censored(
            y_inter_rep_str['Event'],
            y_inter_rep_str['Progression Free Survival'],
            pred_rep
        )[0]

        # Uno C-index: y_boot as training set (correct censoring distribution),
        # TAU fixed globally so truncation is consistent across all iterations.
        try:
            bs_uno_disc[i] = concordance_index_ipcw(
                y_boot, y_inter_disc_str, pred_disc, tau=TAU)[0]
        except Exception:
            pass

        try:
            bs_uno_rep[i] = concordance_index_ipcw(
                y_boot, y_inter_rep_str, pred_rep, tau=TAU)[0]
        except Exception:
            pass

    except Exception:
        bs_pred_disc[i, :]    = np.nan
        bs_pred_rep[i, :]     = np.nan
        bs_surv_disc[i, :, :] = np.nan
        bs_surv_rep[i, :, :]  = np.nan
        bs_coefs[i, :]        = np.nan

successful_M1 = int(np.sum(~np.isnan(bs_har_disc)))
uno_disc_n_M1 = int(np.sum(~np.isnan(bs_uno_disc)))
uno_rep_n_M1  = int(np.sum(~np.isnan(bs_uno_rep)))
print(f"\nSuccessful iterations: {successful_M1}/{N_BOOTSTRAP}")
print(f"Uno successful: Discovery={uno_disc_n_M1}, Replicate={uno_rep_n_M1}")

feature_stability = pd.DataFrame({
    'Feature':              selected_feature_names,
    'Original_Coef':        original_coefs_resnet[selected_feature_indices],
    'Bootstrap_Mean_Coef':  np.nanmean(bs_coefs, axis=0),
    'Bootstrap_Std_Coef':   np.nanstd(bs_coefs,  axis=0),
    'Bootstrap_CI_2.5th':   np.nanpercentile(bs_coefs, 2.5,  axis=0),
    'Bootstrap_CI_97.5th':  np.nanpercentile(bs_coefs, 97.5, axis=0),
}).sort_values('Bootstrap_Std_Coef', ascending=False)

print(f"\nCoefficient stability (top 5 by variability):")
print(feature_stability.head(5)[['Feature', 'Original_Coef',
                                   'Bootstrap_Mean_Coef', 'Bootstrap_Std_Coef']].to_string(index=False))

M1_df_disc = build_patient_uncertainty(bs_pred_disc, bs_surv_disc,
                                        common_discovery_ids, Y_inter_disc,
                                        'Discovery', 'M1')
M1_df_rep  = build_patient_uncertainty(bs_pred_rep,  bs_surv_rep,
                                        common_replicate_ids, Y_inter_rep,
                                        'Replicate',  'M1')
M1_df_all  = pd.concat([M1_df_disc, M1_df_rep], ignore_index=True)

har_disc_v  = bs_har_disc[~np.isnan(bs_har_disc)]
har_rep_v   = bs_har_rep[~np.isnan(bs_har_rep)]
uno_disc_v  = bs_uno_disc[~np.isnan(bs_uno_disc)]
uno_rep_v   = bs_uno_rep[~np.isnan(bs_uno_rep)]

print("\nM1 Results (evaluated on intersection):")
print(f"  Discovery Harrell C : {har_disc_v.mean():.4f} "
      f"[{np.percentile(har_disc_v, 2.5):.4f}, {np.percentile(har_disc_v, 97.5):.4f}]")
print(f"  Replicate Harrell C : {har_rep_v.mean():.4f} "
      f"[{np.percentile(har_rep_v, 2.5):.4f}, {np.percentile(har_rep_v, 97.5):.4f}]")
print(f"  Discovery Uno C     : {uno_disc_v.mean():.4f} "
      f"[{np.percentile(uno_disc_v, 2.5):.4f}, {np.percentile(uno_disc_v, 97.5):.4f}]  "
      f"(n={len(uno_disc_v)})")
print(f"  Replicate Uno C     : {uno_rep_v.mean():.4f} "
      f"[{np.percentile(uno_rep_v, 2.5):.4f}, {np.percentile(uno_rep_v, 97.5):.4f}]  "
      f"(n={len(uno_rep_v)})")
print(f"  Discovery Risk CI Width: {M1_df_disc['Risk_CI_Width'].mean():.4f}")
print(f"  Replicate Risk CI Width: {M1_df_rep['Risk_CI_Width'].mean():.4f}")

M1_df_all.to_csv(os.path.join(OUTPUT_DIR, 'M1_patient_uncertainty.csv'), index=False)

c_index_df = pd.DataFrame({
    'Iteration':         range(N_BOOTSTRAP),
    'Model':             'M1',
    'Harrell_Discovery': bs_har_disc,
    'Harrell_Replicate': bs_har_rep,
    'Uno_Discovery':     bs_uno_disc,
    'Uno_Replicate':     bs_uno_rep,
})
c_index_df.to_csv(os.path.join(OUTPUT_DIR, 'M1_c_index_distributions.csv'), index=False)
pd.DataFrame(bs_coefs, columns=selected_feature_names).to_csv(
    os.path.join(OUTPUT_DIR, 'M1_coefficient_distributions.csv'), index=False)
feature_stability.to_csv(os.path.join(OUTPUT_DIR, 'M1_feature_stability.csv'), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# M2 BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("M2: BOOTSTRAP UNCERTAINTY ANALYSIS (Clinico-ResNet + Molecular)")
print(f"  Bootstrap pool : {n_inter_disc} intersection discovery patients")
print(f"  Evaluation on  : {n_inter_disc} intersection discovery + {n_inter_rep} replicate")
print(f"  Scalers refit on each bootstrap sample for normalization consistency")
print("=" * 80)

resnet_disc_risks    = pd.read_csv(DLM1_DIR / 'outputs' / 'data' / 'discovery_results.csv', index_col='SubjectID')
resnet_rep_risks     = pd.read_csv(DLM1_DIR / 'outputs' / 'data' / 'replicate_results.csv', index_col='SubjectID')
molecular_disc_risks = pd.read_csv(MOL_DIR  / 'outputs' / 'data' / 'discovery_results.csv', index_col='SubjectID')
molecular_rep_risks  = pd.read_csv(MOL_DIR  / 'outputs' / 'data' / 'replicate_results.csv', index_col='SubjectID')

cr_disc_raw = resnet_disc_risks.loc[common_discovery_ids, 'Risk Score'].values
cr_rep_raw  = resnet_rep_risks.loc[common_replicate_ids,  'Risk Score'].values
mo_disc_raw = molecular_disc_risks.loc[common_discovery_ids, 'Risk Score'].values
mo_rep_raw  = molecular_rep_risks.loc[common_replicate_ids,  'Risk Score'].values

M2_bs_pred_disc = np.zeros((N_BOOTSTRAP, n_inter_disc))
M2_bs_pred_rep  = np.zeros((N_BOOTSTRAP, n_inter_rep))
M2_bs_surv_disc = np.zeros((N_BOOTSTRAP, n_inter_disc, len(TIME_POINTS)))
M2_bs_surv_rep  = np.zeros((N_BOOTSTRAP, n_inter_rep,  len(TIME_POINTS)))
M2_bs_coefs     = np.zeros((N_BOOTSTRAP, 2))
M2_bs_har_disc  = np.full(N_BOOTSTRAP, np.nan)
M2_bs_har_rep   = np.full(N_BOOTSTRAP, np.nan)
M2_bs_uno_disc  = np.full(N_BOOTSTRAP, np.nan)
M2_bs_uno_rep   = np.full(N_BOOTSTRAP, np.nan)

print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations...")

for i in tqdm(range(N_BOOTSTRAP), desc="M2 Bootstrap"):
    boot_idx = M2_boot_indices[i]

    cr_boot = cr_disc_raw[boot_idx]
    mo_boot = mo_disc_raw[boot_idx]

    scaler_cr_b = StandardScaler().fit(cr_boot.reshape(-1, 1))
    scaler_mo_b = StandardScaler().fit(mo_boot.reshape(-1, 1))

    X_boot = np.column_stack([
        scaler_cr_b.transform(cr_boot.reshape(-1, 1)).flatten(),
        scaler_mo_b.transform(mo_boot.reshape(-1, 1)).flatten(),
    ])

    X_eval_disc = np.column_stack([
        scaler_cr_b.transform(cr_disc_raw.reshape(-1, 1)).flatten(),
        scaler_mo_b.transform(mo_disc_raw.reshape(-1, 1)).flatten(),
    ])
    X_eval_rep = np.column_stack([
        scaler_cr_b.transform(cr_rep_raw.reshape(-1, 1)).flatten(),
        scaler_mo_b.transform(mo_rep_raw.reshape(-1, 1)).flatten(),
    ])

    y_boot = y_inter_disc_str[boot_idx]

    try:
        m = CoxPHSurvivalAnalysis(alpha=0.0, verbose=0)
        m.fit(X_boot, y_boot)

        M2_bs_coefs[i, :] = m.coef_

        pred_disc = m.predict(X_eval_disc)
        pred_rep  = m.predict(X_eval_rep)
        M2_bs_pred_disc[i, :] = pred_disc
        M2_bs_pred_rep[i, :]  = pred_rep

        M2_bs_surv_disc[i, :, :] = extract_survival_at_times(
            m.predict_survival_function(X_eval_disc), TIME_POINTS)
        M2_bs_surv_rep[i, :, :]  = extract_survival_at_times(
            m.predict_survival_function(X_eval_rep),  TIME_POINTS)

        M2_bs_har_disc[i] = concordance_index_censored(
            y_inter_disc_str['Event'],
            y_inter_disc_str['Progression Free Survival'],
            pred_disc
        )[0]
        M2_bs_har_rep[i] = concordance_index_censored(
            y_inter_rep_str['Event'],
            y_inter_rep_str['Progression Free Survival'],
            pred_rep
        )[0]

        # Uno C-index: y_boot as training set (correct censoring distribution),
        # TAU fixed globally so truncation is consistent across all iterations.
        try:
            M2_bs_uno_disc[i] = concordance_index_ipcw(
                y_boot, y_inter_disc_str, pred_disc, tau=TAU)[0]
        except Exception:
            pass

        try:
            M2_bs_uno_rep[i] = concordance_index_ipcw(
                y_boot, y_inter_rep_str, pred_rep, tau=TAU)[0]
        except Exception:
            pass

    except Exception:
        M2_bs_pred_disc[i, :]    = np.nan
        M2_bs_pred_rep[i, :]     = np.nan
        M2_bs_surv_disc[i, :, :] = np.nan
        M2_bs_surv_rep[i, :, :]  = np.nan
        M2_bs_coefs[i, :]        = np.nan

successful_M2 = int(np.sum(~np.isnan(M2_bs_har_disc)))
uno_disc_n_M2 = int(np.sum(~np.isnan(M2_bs_uno_disc)))
uno_rep_n_M2  = int(np.sum(~np.isnan(M2_bs_uno_rep)))
print(f"\nSuccessful iterations: {successful_M2}/{N_BOOTSTRAP}")
print(f"Uno successful: Discovery={uno_disc_n_M2}, Replicate={uno_rep_n_M2}")

M2_df_disc = build_patient_uncertainty(M2_bs_pred_disc, M2_bs_surv_disc,
                                        common_discovery_ids, Y_inter_disc,
                                        'Discovery', 'M2')
M2_df_rep  = build_patient_uncertainty(M2_bs_pred_rep,  M2_bs_surv_rep,
                                        common_replicate_ids, Y_inter_rep,
                                        'Replicate',  'M2')
M2_df_all  = pd.concat([M2_df_disc, M2_df_rep], ignore_index=True)

har_disc_v2 = M2_bs_har_disc[~np.isnan(M2_bs_har_disc)]
har_rep_v2  = M2_bs_har_rep[~np.isnan(M2_bs_har_rep)]
uno_disc_v2 = M2_bs_uno_disc[~np.isnan(M2_bs_uno_disc)]
uno_rep_v2  = M2_bs_uno_rep[~np.isnan(M2_bs_uno_rep)]

print("\nM2 Results (evaluated on intersection):")
print(f"  Discovery Harrell C : {har_disc_v2.mean():.4f} "
      f"[{np.percentile(har_disc_v2, 2.5):.4f}, {np.percentile(har_disc_v2, 97.5):.4f}]")
print(f"  Replicate Harrell C : {har_rep_v2.mean():.4f} "
      f"[{np.percentile(har_rep_v2, 2.5):.4f}, {np.percentile(har_rep_v2, 97.5):.4f}]")
print(f"  Discovery Uno C     : {uno_disc_v2.mean():.4f} "
      f"[{np.percentile(uno_disc_v2, 2.5):.4f}, {np.percentile(uno_disc_v2, 97.5):.4f}]  "
      f"(n={len(uno_disc_v2)})")
print(f"  Replicate Uno C     : {uno_rep_v2.mean():.4f} "
      f"[{np.percentile(uno_rep_v2, 2.5):.4f}, {np.percentile(uno_rep_v2, 97.5):.4f}]  "
      f"(n={len(uno_rep_v2)})")
print(f"  Discovery Risk CI Width: {M2_df_disc['Risk_CI_Width'].mean():.4f}")
print(f"  Replicate Risk CI Width: {M2_df_rep['Risk_CI_Width'].mean():.4f}")
print(f"\nFusion Coefficients (bootstrap mean):")
print(f"  ClinResNet: {np.nanmean(M2_bs_coefs[:, 0]):.4f} ± {np.nanstd(M2_bs_coefs[:, 0]):.4f}")
print(f"  Molecular:  {np.nanmean(M2_bs_coefs[:, 1]):.4f} ± {np.nanstd(M2_bs_coefs[:, 1]):.4f}")

M2_df_all.to_csv(os.path.join(OUTPUT_DIR, 'M2_patient_uncertainty.csv'), index=False)

c_index_df_M2 = pd.DataFrame({
    'Iteration':         range(N_BOOTSTRAP),
    'Model':             'M2',
    'Harrell_Discovery': M2_bs_har_disc,
    'Harrell_Replicate': M2_bs_har_rep,
    'Uno_Discovery':     M2_bs_uno_disc,
    'Uno_Replicate':     M2_bs_uno_rep,
})
c_index_df_M2.to_csv(os.path.join(OUTPUT_DIR, 'M2_c_index_distributions.csv'), index=False)
pd.DataFrame({'ClinResNet_Coef': M2_bs_coefs[:, 0],
              'Molecular_Coef':  M2_bs_coefs[:, 1]}).to_csv(
    os.path.join(OUTPUT_DIR, 'M2_fusion_coefficient_distributions.csv'), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("COMPARISON ANALYSIS: M1 vs M2 (Clinico-ResNet)")
print("=" * 80)

summary_M1  = compute_summary_stats(M1_df_all, 'M1')
summary_M2  = compute_summary_stats(M2_df_all, 'M2')
summary_all = pd.concat([summary_M1, summary_M2], ignore_index=True)

merged = M1_df_all[['SubjectID', 'Cohort', 'Risk_CI_Width', 'Surv_60m_CI_Width', 'PFS', 'Event']].copy()
merged = merged.rename(columns={'Risk_CI_Width':     'M1_Risk_CI_Width',
                                 'Surv_60m_CI_Width': 'M1_Surv60_CI_Width'})
merged = merged.merge(
    M2_df_all[['SubjectID', 'Risk_CI_Width', 'Surv_60m_CI_Width']].rename(
        columns={'Risk_CI_Width':     'M2_Risk_CI_Width',
                 'Surv_60m_CI_Width': 'M2_Surv60_CI_Width'}),
    on='SubjectID', how='inner'
)

merged['Delta_M1_M2_Risk']         = merged['M1_Risk_CI_Width'] - merged['M2_Risk_CI_Width']
merged['Delta_M1_M2_Surv60']       = merged['M1_Surv60_CI_Width'] - merged['M2_Surv60_CI_Width']
merged['Pct_Reduction_M1_M2_Risk'] = (merged['Delta_M1_M2_Risk'] / merged['M1_Risk_CI_Width']) * 100

print("\nStatistical Tests (Replicate Cohort):")
rep_data = merged[merged['Cohort'] == 'Replicate']
stat, p_m1_m2 = stats.wilcoxon(rep_data['M1_Risk_CI_Width'], rep_data['M2_Risk_CI_Width'])
print(f"  Wilcoxon Signed-Rank (Risk CI Width) M1 vs M2: p = {p_m1_m2:.4f}")

summary_all.to_csv(os.path.join(OUTPUT_DIR, 'COMPARISON_summary_statistics.csv'), index=False)
merged.to_csv(os.path.join(OUTPUT_DIR, 'COMPARISON_patient_level.csv'), index=False)

c_summary_M1  = compute_c_index_summary(c_index_df,    'M1')
c_summary_M2  = compute_c_index_summary(c_index_df_M2, 'M2')
c_summary_all = pd.concat([c_summary_M1, c_summary_M2], ignore_index=True)
c_summary_all.to_csv(os.path.join(OUTPUT_DIR, 'COMPARISON_c_index_statistics.csv'), index=False)

pub_rows = []
for cohort in ['Discovery', 'Replicate']:
    for _, row in summary_all[summary_all['Cohort'] == cohort].iterrows():
        pub_rows.append({
            'Model': row['Model'], 'Cohort': cohort, 'N': row['N_Patients'],
            'Risk_CI_Width_Median_IQR':   f"{row['Risk_CI_Width_Median']:.3f} "
                                           f"({row['Risk_CI_Width_Q25']:.3f}–{row['Risk_CI_Width_Q75']:.3f})",
            'Surv60_CI_Width_Median_IQR': f"{row['Surv60_CI_Width_Median']:.3f} "
                                           f"({row['Surv60_CI_Width_Q25']:.3f}–{row['Surv60_CI_Width_Q75']:.3f})",
        })
pd.DataFrame(pub_rows).to_csv(
    os.path.join(OUTPUT_DIR, 'PUBLICATION_Table_Uncertainty_Comparison.csv'), index=False)

combined_pub = c_summary_all.merge(summary_all, on=['Model', 'Cohort'], how='inner')
combined_pub['Harrell_C']        = combined_pub.apply(
    lambda x: f"{x['Harrell_Mean']:.3f} [{x['Harrell_CI_2.5th']:.3f}, {x['Harrell_CI_97.5th']:.3f}]", axis=1)
combined_pub['Uno_C']            = combined_pub.apply(
    lambda x: f"{x['Uno_Mean']:.3f} [{x['Uno_CI_2.5th']:.3f}, {x['Uno_CI_97.5th']:.3f}]", axis=1)
combined_pub['Risk_Uncertainty'] = combined_pub.apply(
    lambda x: f"{x['Risk_CI_Width_Median']:.3f} (IQR: {x['Risk_CI_Width_IQR']:.3f})", axis=1)
combined_pub[['Model', 'Cohort', 'N_Patients', 'Harrell_C', 'Uno_C', 'Risk_Uncertainty']].to_csv(
    os.path.join(OUTPUT_DIR, 'PUBLICATION_Table_Combined_Performance_Uncertainty.csv'), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# PAIRED BOOTSTRAP SIGNIFICANCE TESTING: M2 vs M1
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PAIRED BOOTSTRAP SIGNIFICANCE: M2 vs M1")
print("=" * 80)

sig_results = []

for cohort_type in ['Discovery', 'Replicate']:

    har_m1 = c_index_df[f'Harrell_{cohort_type}'].values
    har_m2 = c_index_df_M2[f'Harrell_{cohort_type}'].values
    uno_m1 = c_index_df[f'Uno_{cohort_type}'].values
    uno_m2 = c_index_df_M2[f'Uno_{cohort_type}'].values

    for metric_name, m1_vals, m2_vals in [
            ('Harrell', har_m1, har_m2),
            ('Uno',     uno_m1, uno_m2)]:

        # Keep only iterations where BOTH models succeeded
        valid = ~np.isnan(m1_vals) & ~np.isnan(m2_vals)
        diff  = (m2_vals - m1_vals)[valid]   # positive = M2 better
        n     = valid.sum()

        if n < 10:
            print(f"  {cohort_type} {metric_name}: insufficient paired iterations (n={n})")
            continue

        mean_diff   = diff.mean()
        ci_lo       = np.percentile(diff, 2.5)
        ci_hi       = np.percentile(diff, 97.5)
        # Two-sided p-value: proportion of bootstrap differences on the wrong side
        p_val       = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
        significant = "YES" if ci_lo > 0 else "NO"

        print(f"\n  {cohort_type} | {metric_name} C-index:")
        print(f"    M1 mean : {np.nanmean(m1_vals):.4f}")
        print(f"    M2 mean : {np.nanmean(m2_vals):.4f}")
        print(f"    Δ (M2−M1): {mean_diff:+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"    Two-sided p = {p_val:.4f}  |  Significant: {significant}")
        print(f"    (based on {n} paired iterations)")

        sig_results.append({
            'Cohort':           cohort_type,
            'Metric':           metric_name,
            'N_Paired':         n,
            'M1_Mean':          np.nanmean(m1_vals),
            'M2_Mean':          np.nanmean(m2_vals),
            'Delta_Mean':       mean_diff,
            'Delta_CI_2.5th':   ci_lo,
            'Delta_CI_97.5th':  ci_hi,
            'Delta_CI_Width':   ci_hi - ci_lo,
            'P_Value':          p_val,
            'Significant':      significant,
        })

sig_df = pd.DataFrame(sig_results)
sig_df.to_csv(os.path.join(OUTPUT_DIR, 'SIGNIFICANCE_paired_bootstrap_M2_vs_M1.csv'), index=False)

# Publication-ready table
print("\n  Publication summary:")
print(f"  {'Cohort':<12} {'Metric':<10} {'Δ (M2−M1)':<14} {'95% CI':<26} {'p-value':<10} {'Sig'}")
print("  " + "-" * 78)
for _, r in sig_df.iterrows():
    ci_str = f"[{r['Delta_CI_2.5th']:+.3f}, {r['Delta_CI_97.5th']:+.3f}]"
    print(f"  {r['Cohort']:<12} {r['Metric']:<10} {r['Delta_Mean']:+.4f}{'':>8} "
          f"{ci_str:<26} {r['P_Value']:.4f}{'':>4} {r['Significant']}")

print(f"\n  Saved: SIGNIFICANCE_paired_bootstrap_M2_vs_M1.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("GENERATING FIGURES")
print("=" * 80)

merged_disc = merged[merged['Cohort'] == 'Discovery']
merged_rep  = merged[merged['Cohort'] == 'Replicate']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: Risk CI Width box plots
ax = axes[0, 0]
data_to_plot, labels, colors = [], [], []
for cohort, cdata in [('Discovery', merged_disc), ('Replicate', merged_rep)]:
    for model in ['M1', 'M2']:
        data_to_plot.append(cdata[f'{model}_Risk_CI_Width'].values)
        labels.append(f'{model}\n{cohort}')
        colors.append({'M1': '#e74c3c', 'M2': '#3498db'}[model])
bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel('Risk Score CI Width (95% CI)', fontsize=12, fontweight='bold')
ax.set_title('A) Risk Score Uncertainty', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Panel B: Survival CI Width box plots at 60 months
ax = axes[0, 1]
data_to_plot, labels, colors = [], [], []
for cohort, cdata in [('Discovery', merged_disc), ('Replicate', merged_rep)]:
    for model in ['M1', 'M2']:
        data_to_plot.append(cdata[f'{model}_Surv60_CI_Width'].values)
        labels.append(f'{model}\n{cohort}')
        colors.append({'M1': '#e74c3c', 'M2': '#3498db'}[model])
bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel('Survival Probability CI Width at 60m (95% CI)', fontsize=12, fontweight='bold')
ax.set_title('B) Survival Probability Uncertainty at 60 Months', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Panel C: Delta violin plots with stats
ax = axes[1, 0]
violin_data = []
for cohort, cdata, color in [
        ('Discovery', merged_disc, '#3498db'),
        ('Replicate',  merged_rep,  '#e74c3c')]:
    delta = cdata['Delta_M1_M2_Risk'].values
    violin_data.append((cohort, cdata, color, delta))
    pos = len(violin_data) - 1
    pcts = ax.violinplot([delta], positions=[pos],
                         showmeans=True, showmedians=True, widths=0.7)
    for pc in pcts['bodies']:
        pc.set_facecolor(color); pc.set_alpha(0.6)

ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Discovery', 'Replicate'], fontsize=11)
ax.set_ylabel('Δ Risk CI Width (M1 − M2)', fontsize=12, fontweight='bold')
ax.set_title('C) Uncertainty Reduction: M1 → M2\n(Positive = M2 is tighter)',
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Capture y_max after all violins are drawn
y_max = ax.get_ylim()[1]
offsets = [0.38, -0.34]

for (pos, (cohort, cdata, color, delta)), x_off in zip(enumerate(violin_data), offsets):
    m1_ci = cdata['M1_Risk_CI_Width'].values
    m2_ci = cdata['M2_Risk_CI_Width'].values
    _, p  = stats.wilcoxon(m1_ci, m2_ci)
    pooled_std = np.sqrt((np.std(m1_ci, ddof=1)**2 + np.std(m2_ci, ddof=1)**2) / 2)
    d = (np.mean(m1_ci) - np.mean(m2_ci)) / pooled_std
    p_str = ("p<0.001" if p < 0.001 else
             "p<0.01"  if p < 0.01  else
             "p<0.05"  if p < 0.05  else f"p={p:.3f}")
    ax.text(pos + x_off, y_max * 0.90,
            f"Median Δ: {np.median(delta):.3f}\n"
            f"Relative: {np.median(cdata['Pct_Reduction_M1_M2_Risk']):.1f}%\n"
            f"{p_str}\nCohen's d: {d:.2f}",
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                      edgecolor=color, linewidth=2))

# Panel D: Uncertainty vs PFS scatter (Replicate)
ax = axes[1, 1]
for model, color, marker in [('M1', '#e74c3c', 'o'), ('M2', '#3498db', 's')]:
    ax.scatter(merged_rep['PFS'], merged_rep[f'{model}_Risk_CI_Width'],
               c=color, alpha=0.6, s=80, marker=marker, label=model,
               edgecolors='black', linewidth=0.5)
ax.set_xlabel('Progression-Free Survival (months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Risk Score CI Width (95% CI)', fontsize=12, fontweight='bold')
ax.set_title('D) Uncertainty vs Follow-up Time (Replicate)', fontsize=14, fontweight='bold')
ax.legend(title='Model', fontsize=10)
ax.grid(alpha=0.3)

fig.suptitle(
    'Uncertainty Comparison: M1 (Clinico-ResNet) vs '
    'M2 (Clinico-ResNet + Molecular)\n',
    fontsize=16, fontweight='bold', y=0.995)
fig.text(
    0.5, -0.02,
    "Box plots show the distribution of per-patient 95% bootstrap CI widths across 1,000 iterations.",
    ha='center', va='bottom', fontsize=14, fontweight='bold', color='#111111',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#888888', linewidth=1)
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'FIGURE_Uncertainty_Reduction_Overview.jpg'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'FIGURE_Uncertainty_Reduction_Overview.pdf'),
            bbox_inches='tight')
plt.close()

print("  Figure saved: FIGURE_Uncertainty_Reduction_Overview.jpg/pdf")
print(f"\nAll results saved to: {OUTPUT_DIR}")
print("\nAnalysis complete.")