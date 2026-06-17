"""
Permutation Sanity Check for M2 CI Width Reduction
====================================================
Reviewer concern: A 75% reduction in bootstrap CI width when adding a
categorical molecular feature could be a mechanical statistical artifact
from adding any well-populated covariate, rather than a genuine epistemic
gain from molecular biology.

Test (Option A — Permuted Molecular Labels):
  Randomly shuffle the molecular subtype labels across patients, breaking
  the biology-prediction link while preserving the statistical structure
  of the feature (same cardinality, same category frequencies). Run the
  full M2-style late fusion bootstrap with the permuted labels and record
  the CI width reduction relative to M1. Repeat N_PERMUTATIONS times.

  If the real reduction is driven by biology, it should be substantially
  larger than the permutation distribution. If permutation gives similar
  reductions, the reviewer's concern is confirmed.

REAL M2 BASELINE:
  The real M2 CI widths are computed using the SAME molecular risk scores
  as the original uncertainty script — loaded directly from the saved
  molecular model's output CSVs (outputs/data/discovery_results.csv,
  replicate_results.csv). This ensures the "real reduction" being tested
  matches exactly the figures reported in the paper.

CI WIDTH REDUCTION METRIC:
  All reductions are computed on the MEDIAN CI width per cohort, consistent
  with the publication table (Risk_CI_Width_Median). The CI width
  distribution is right-skewed, so median is more robust than mean and
  directly matches the reported numbers.

PERMUTATION PROCEDURE — mirrors M1 feature-selection approach:
  The saved molecular CoxnetSurvivalAnalysis estimator identifies which
  features were selected (nonzero coefficients). A plain CoxPH is then
  refit on those selected features with PERMUTED outcome labels, mirroring
  how M1 refits plain CoxPH on its LASSO-selected features. The selected
  feature set is fixed across all permutations — only the outcomes change.

Author: Fadel Batal
Date: June 2026
"""

import pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

N_PERMUTATIONS = 500    # number of permuted-label runs
N_BOOTSTRAP    = 1000   # bootstrap iterations per permutation — matches original M2
RANDOM_SEED    = 42

# Repo root: one level up from this script's folder (04_Analysis/)
REPO_ROOT      = Path(__file__).resolve().parent.parent
DLM1_DIR       = REPO_ROOT / '01_DLM1_Clinico_ResNet'
MOL_DIR        = REPO_ROOT / '02_Molecular_Subtype'

OUTPUT_DIR = str(REPO_ROOT / '04_Analysis' / 'outputs' / 'permutation_sanity_check')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def convert_to_structured_array(y_df):
    return np.array(
        [(bool(e), float(t)) for e, t in
         zip(y_df['Event'], y_df['Progression Free Survival'])],
        dtype=[('Event', bool), ('Progression Free Survival', float)]
    )


def ci_widths(preds):
    """Per-patient 95% CI width across bootstrap predictions."""
    return np.array([
        np.percentile(col[~np.isnan(col)], 97.5) -
        np.percentile(col[~np.isnan(col)], 2.5)
        for col in preds.T
    ])


def median_ci_reduction(m1_widths, m2_widths):
    """
    Median of per-patient percent reductions, matching the publication
    metric (median of Pct_Reduction_M1_M2_Risk in the uncertainty script).
    """
    pct = (m1_widths - m2_widths) / m1_widths * 100
    return float(np.median(pct))


def run_M2_bootstrap(cr_disc_raw, mo_disc_raw, cr_rep_raw, mo_rep_raw,
                     y_inter_disc_str, y_inter_rep_str,
                     n_inter_disc, boot_indices):
    """
    Run the M2 bootstrap exactly as in the original uncertainty script.
    Scalers are refit on each bootstrap resample for consistency.
    Returns per-patient risk CI widths for discovery and replicate.
    """
    n_rep        = len(cr_rep_raw)
    bs_pred_disc = np.zeros((N_BOOTSTRAP, n_inter_disc))
    bs_pred_rep  = np.zeros((N_BOOTSTRAP, n_rep))

    for i, boot_idx in enumerate(boot_indices):
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
            bs_pred_disc[i, :] = m.predict(X_eval_disc)
            bs_pred_rep[i, :]  = m.predict(X_eval_rep)
        except Exception:
            bs_pred_disc[i, :] = np.nan
            bs_pred_rep[i, :]  = np.nan

    return ci_widths(bs_pred_disc), ci_widths(bs_pred_rep)


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("=" * 80)
print("PERMUTATION SANITY CHECK — M2 CI WIDTH REDUCTION")
print("=" * 80)
print("\nStep 1: Loading data...")

with open(DLM1_DIR / 'outputs' / 'models' / 'X_clinical.pkl', 'rb') as f:
    X_clinical = pickle.load(f)
with open(DLM1_DIR / 'outputs' / 'models' / 'X_resnet.pkl', 'rb') as f:
    X_resnet = pickle.load(f)
with open(DLM1_DIR / 'outputs' / 'models' / 'y.pkl', 'rb') as f:
    Y_resnet = pickle.load(f)
with open(DLM1_DIR / 'outputs' / 'models' / 'estimator.pkl', 'rb') as f:
    trained_model_resnet = pickle.load(f)

with open(MOL_DIR / 'outputs' / 'models' / 'estimator.pkl', 'rb') as f:
    mol_estimator_saved = pickle.load(f)

X_clinical_features   = X_clinical.drop(columns=['Cohort'], errors='ignore')
X_resnet_features     = X_resnet.drop(columns=['Cohort'],   errors='ignore')
X_clinresnet_combined = pd.concat([X_clinical_features, X_resnet_features], axis=1).sort_index()
Y_resnet              = Y_resnet.sort_index()

X_molecular = pd.read_pickle(MOL_DIR / 'outputs' / 'models' / 'X_molecular.pkl').sort_index()

print(f"  Clinical-ResNet: {X_clinresnet_combined.shape}")
print(f"  Molecular:       {X_molecular.shape}")

# =============================================================================
# STEP 2: BUILD PATIENT SETS AND EXTRACT SELECTED FEATURES
# =============================================================================

print("\nStep 2: Building patient sets and extracting selected features...")

cr_coefs             = trained_model_resnet.coef_[:, 0]
cr_selected_features = X_clinresnet_combined.columns[cr_coefs != 0].tolist()
print(f"  M1 selected features: {len(cr_selected_features)} / {len(cr_coefs)}")

X_mol_all_cols   = X_molecular.drop(columns=['Cohort'], errors='ignore').columns.tolist()
mol_coefs        = np.asarray(mol_estimator_saved.coef_).reshape(-1)
mol_sel_mask     = np.abs(mol_coefs) > 1e-12
mol_sel_features = [c for c, s in zip(X_mol_all_cols, mol_sel_mask) if s]
print(f"  Molecular selected features: {len(mol_sel_features)} / {len(mol_coefs)}")
print(f"  Selected: {mol_sel_features}")

common_all_ids       = sorted(set(X_clinresnet_combined.index) & set(X_molecular.index))
common_discovery_ids = [p for p in common_all_ids if Y_resnet.loc[p, 'Cohort'] == 'Discovery']
common_replicate_ids = [p for p in common_all_ids if Y_resnet.loc[p, 'Cohort'] == 'Replicate']
n_inter_disc         = len(common_discovery_ids)
n_inter_rep          = len(common_replicate_ids)
print(f"  Intersection discovery: {n_inter_disc}  replicate: {n_inter_rep}")

Y_inter_disc     = Y_resnet.loc[common_discovery_ids]
Y_inter_rep      = Y_resnet.loc[common_replicate_ids]
y_inter_disc_str = convert_to_structured_array(Y_inter_disc)
y_inter_rep_str  = convert_to_structured_array(Y_inter_rep)

X_inter_disc_M1 = X_clinresnet_combined.loc[common_discovery_ids, cr_selected_features].values
X_inter_rep_M1  = X_clinresnet_combined.loc[common_replicate_ids,  cr_selected_features].values

X_mol_base      = X_molecular.drop(columns=['Cohort'], errors='ignore')
X_mol_discovery = X_mol_base.loc[common_discovery_ids, mol_sel_features]
X_mol_replicate = X_mol_base.loc[common_replicate_ids, mol_sel_features]

zero_var = [c for c in mol_sel_features if X_mol_discovery[c].std() == 0]
if zero_var:
    print(f"  WARNING: dropping zero-variance mol columns in intersection: {zero_var}")
    mol_sel_features = [c for c in mol_sel_features if c not in zero_var]
    X_mol_discovery  = X_mol_discovery[mol_sel_features]
    X_mol_replicate  = X_mol_replicate[mol_sel_features]

# =============================================================================
# STEP 2b: LOAD RAW RISK SCORES FROM SAVED CSVs
# =============================================================================

resnet_disc_risks    = pd.read_csv(DLM1_DIR / 'outputs' / 'data' / 'discovery_results.csv', index_col='SubjectID')
resnet_rep_risks     = pd.read_csv(DLM1_DIR / 'outputs' / 'data' / 'replicate_results.csv', index_col='SubjectID')
molecular_disc_risks = pd.read_csv(MOL_DIR  / 'outputs' / 'data' / 'discovery_results.csv', index_col='SubjectID')
molecular_rep_risks  = pd.read_csv(MOL_DIR  / 'outputs' / 'data' / 'replicate_results.csv', index_col='SubjectID')

cr_disc_raw = resnet_disc_risks.loc[common_discovery_ids, 'Risk Score'].values
cr_rep_raw  = resnet_rep_risks.loc[common_replicate_ids,  'Risk Score'].values
mo_disc_raw = molecular_disc_risks.loc[common_discovery_ids, 'Risk Score'].values
mo_rep_raw  = molecular_rep_risks.loc[common_replicate_ids,  'Risk Score'].values

print(f"  CR risk scores loaded  — discovery: {len(cr_disc_raw)}  replicate: {len(cr_rep_raw)}")
print(f"  Mol risk scores loaded — discovery: {len(mo_disc_raw)}  replicate: {len(mo_rep_raw)}")

# =============================================================================
# STEP 3: COMPUTE M1 CI WIDTHS (reference, run once)
# =============================================================================

print("\nStep 3: Computing M1 CI widths (reference, 1000 bootstrap iterations)...")

full_discovery_ids = sorted(Y_resnet[Y_resnet['Cohort'] == 'Discovery'].index)
n_full_disc        = len(full_discovery_ids)
Y_full_disc        = Y_resnet.loc[full_discovery_ids]
X_full_disc        = X_clinresnet_combined.loc[full_discovery_ids, cr_selected_features].values
y_full_disc_str    = convert_to_structured_array(Y_full_disc)

rng_m1 = np.random.RandomState(RANDOM_SEED)
M1_boot_indices = [
    rng_m1.choice(n_full_disc, size=n_full_disc, replace=True)
    for _ in range(N_BOOTSTRAP)
]

m1_bs_pred_disc = np.zeros((N_BOOTSTRAP, n_inter_disc))
m1_bs_pred_rep  = np.zeros((N_BOOTSTRAP, n_inter_rep))

for i, boot_idx in enumerate(tqdm(M1_boot_indices, desc="M1 Bootstrap")):
    try:
        m = CoxPHSurvivalAnalysis(alpha=0.0, verbose=0)
        m.fit(X_full_disc[boot_idx], y_full_disc_str[boot_idx])
        m1_bs_pred_disc[i, :] = m.predict(X_inter_disc_M1)
        m1_bs_pred_rep[i, :]  = m.predict(X_inter_rep_M1)
    except Exception:
        m1_bs_pred_disc[i, :] = np.nan
        m1_bs_pred_rep[i, :]  = np.nan

m1_ci_disc = ci_widths(m1_bs_pred_disc)
m1_ci_rep  = ci_widths(m1_bs_pred_rep)
print(f"  M1 median CI width — Discovery: {np.median(m1_ci_disc):.4f}  Replicate: {np.median(m1_ci_rep):.4f}")

# =============================================================================
# STEP 4: COMPUTE REAL M2 CI WIDTHS (observed reduction)
# =============================================================================

print("\nStep 4: Computing real M2 CI widths (same risk scores as uncertainty script)...")

rng_m2_real = np.random.RandomState(123)
M2_boot_indices = [
    rng_m2_real.choice(n_inter_disc, size=n_inter_disc, replace=True)
    for _ in range(N_BOOTSTRAP)
]

m2_ci_disc_real, m2_ci_rep_real = run_M2_bootstrap(
    cr_disc_raw, mo_disc_raw, cr_rep_raw, mo_rep_raw,
    y_inter_disc_str, y_inter_rep_str,
    n_inter_disc, M2_boot_indices
)

# Reduction based on MEDIAN — consistent with publication table
real_reduction_disc = median_ci_reduction(m1_ci_disc, m2_ci_disc_real)
real_reduction_rep  = median_ci_reduction(m1_ci_rep,  m2_ci_rep_real)

print(f"  M2 (real) median CI width — Discovery: {np.median(m2_ci_disc_real):.4f}  Replicate: {np.median(m2_ci_rep_real):.4f}")
print(f"  Observed CI width reduction (median) — Discovery: {real_reduction_disc:.1f}%  Replicate: {real_reduction_rep:.1f}%")

# =============================================================================
# STEP 5: PERMUTATION LOOP
# =============================================================================

print(f"\nStep 5: Running {N_PERMUTATIONS} permutations of molecular labels...")
print("  Each permutation:")
print("    1. Shuffle outcome labels → refit plain CoxPH on selected mol features")
print("    2. Get permuted molecular risk scores (same selected feature set)")
print("    3. Run full M2 bootstrap (1000 iterations, fresh boot indices)")
print("    4. Record MEDIAN CI width reduction vs M1\n")

perm_reductions_disc = np.full(N_PERMUTATIONS, np.nan)
perm_reductions_rep  = np.full(N_PERMUTATIONS, np.nan)
rng_perm = np.random.RandomState(RANDOM_SEED + 1)

for p in tqdm(range(N_PERMUTATIONS), desc="Permutations"):

    perm_order = rng_perm.permutation(n_inter_disc)
    y_mol_perm = y_inter_disc_str[perm_order]

    try:
        mol_perm = CoxPHSurvivalAnalysis(alpha=0.0, verbose=0)
        mol_perm.fit(X_mol_discovery.values, y_mol_perm)
        mo_disc_perm = mol_perm.predict(X_mol_discovery.values)
        mo_rep_perm  = mol_perm.predict(X_mol_replicate.values)
    except Exception:
        continue

    rng_boot = np.random.RandomState(RANDOM_SEED + 2 + p)
    perm_boot_indices = [
        rng_boot.choice(n_inter_disc, size=n_inter_disc, replace=True)
        for _ in range(N_BOOTSTRAP)
    ]

    perm_ci_disc, perm_ci_rep = run_M2_bootstrap(
        cr_disc_raw, mo_disc_perm, cr_rep_raw, mo_rep_perm,
        y_inter_disc_str, y_inter_rep_str,
        n_inter_disc, perm_boot_indices
    )

    # Reduction based on MEDIAN — consistent with real reduction above
    perm_reductions_disc[p] = median_ci_reduction(m1_ci_disc, perm_ci_disc)
    perm_reductions_rep[p]  = median_ci_reduction(m1_ci_rep,  perm_ci_rep)

valid_disc = perm_reductions_disc[~np.isnan(perm_reductions_disc)]
valid_rep  = perm_reductions_rep[~np.isnan(perm_reductions_rep)]
print(f"\n  Successful permutations: {len(valid_disc)}/{N_PERMUTATIONS}")

# =============================================================================
# STEP 6: RESULTS AND PERMUTATION P-VALUE
# =============================================================================

print("\n" + "=" * 80)
print("PERMUTATION TEST RESULTS")
print("=" * 80)

p_val_disc = float(np.mean(valid_disc >= real_reduction_disc))
p_val_rep  = float(np.mean(valid_rep  >= real_reduction_rep))

for cohort, real, perm_vals, p_val in [
        ('Discovery', real_reduction_disc, valid_disc, p_val_disc),
        ('Replicate',  real_reduction_rep,  valid_rep,  p_val_rep)]:
    print(f"\n  {cohort}:")
    print(f"    Real M2 median CI width reduction:  {real:.1f}%")
    print(f"    Permutation mean reduction:          {perm_vals.mean():.1f}%")
    print(f"    Permutation 95th percentile:         {np.percentile(perm_vals, 95):.1f}%")
    print(f"    Permutation 99th percentile:         {np.percentile(perm_vals, 99):.1f}%")
    print(f"    One-sided permutation p-value:       {p_val:.4f}")
    if p_val < 0.01:
        print(f"    → Real reduction significantly exceeds permutation null (p<0.01).")
        print(f"      CI width reduction is driven by molecular biology signal.")
    elif p_val < 0.05:
        print(f"    → Real reduction exceeds permutation null (p<0.05).")
    else:
        print(f"    → Real reduction NOT significantly different from permutation null.")
        print(f"      Reviewer concern may be warranted — mechanical stabilization likely.")

# =============================================================================
# STEP 7: SAVE RESULTS
# =============================================================================

results_df = pd.DataFrame({
    'permutation':         range(N_PERMUTATIONS),
    'perm_reduction_disc': perm_reductions_disc,
    'perm_reduction_rep':  perm_reductions_rep,
})
results_df.to_csv(os.path.join(OUTPUT_DIR, 'permutation_reductions.csv'), index=False)

summary_df = pd.DataFrame([
    {
        'cohort':                  'Discovery',
        'metric':                  'median CI width reduction',
        'real_reduction_pct':      real_reduction_disc,
        'perm_mean_reduction_pct': valid_disc.mean(),
        'perm_std_reduction_pct':  valid_disc.std(),
        'perm_95th_pct':           np.percentile(valid_disc, 95),
        'perm_99th_pct':           np.percentile(valid_disc, 99),
        'one_sided_p_value':       p_val_disc,
        'n_permutations':          len(valid_disc),
        'significant_p01':         p_val_disc < 0.01,
        'significant_p05':         p_val_disc < 0.05,
        'mol_selected_features':   str(mol_sel_features),
        'real_m2_source':          'saved CSV (outputs/data/discovery_results.csv)',
    },
    {
        'cohort':                  'Replicate',
        'metric':                  'median CI width reduction',
        'real_reduction_pct':      real_reduction_rep,
        'perm_mean_reduction_pct': valid_rep.mean(),
        'perm_std_reduction_pct':  valid_rep.std(),
        'perm_95th_pct':           np.percentile(valid_rep, 95),
        'perm_99th_pct':           np.percentile(valid_rep, 99),
        'one_sided_p_value':       p_val_rep,
        'n_permutations':          len(valid_rep),
        'significant_p01':         p_val_rep < 0.01,
        'significant_p05':         p_val_rep < 0.05,
        'mol_selected_features':   str(mol_sel_features),
        'real_m2_source':          'saved CSV (outputs/data/replicate_results.csv)',
    },
])
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'permutation_summary.csv'), index=False)
print(f"\n  Saved: permutation_reductions.csv, permutation_summary.csv")

# =============================================================================
# STEP 8: FIGURE
# =============================================================================

print("\nGenerating figure...")

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)


def plot_permutation_panel(ax, perm_vals, real_val, p_val, cohort, color):
    ax.hist(perm_vals, bins=40, color=color, alpha=0.65, edgecolor='white',
            linewidth=0.5, label=f'Permuted ({len(perm_vals)} runs)')
    ax.axvline(real_val, color='#c0392b', linewidth=2.5, linestyle='-',
               label=f'Real M2: {real_val:.1f}%')
    ax.axvline(np.percentile(perm_vals, 95), color='#2c3e50', linewidth=1.5,
               linestyle='--', label=f'Perm 95th: {np.percentile(perm_vals, 95):.1f}%')
    p_str = f"p = {p_val:.4f}" if p_val >= 0.001 else "p < 0.001"
    ax.set_xlabel('Median CI Width Reduction vs M1 (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'{cohort} Cohort\n{p_str}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)


ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
plot_permutation_panel(ax_a, valid_disc, real_reduction_disc, p_val_disc, 'Discovery', '#3498db')
plot_permutation_panel(ax_b, valid_rep,  real_reduction_rep,  p_val_rep,  'Replicate',  '#9b59b6')

# Panel C: Median CI widths side by side
ax_c = fig.add_subplot(gs[1, 0])
categories = ['M1\n(baseline)', 'M2 Real\n(true mol.)', 'M2 Perm\n(median perm.)']
disc_vals  = [
    np.median(m1_ci_disc),
    np.median(m2_ci_disc_real),
    np.median(m1_ci_disc) * (1 - np.median(valid_disc) / 100),
]
rep_vals   = [
    np.median(m1_ci_rep),
    np.median(m2_ci_rep_real),
    np.median(m1_ci_rep) * (1 - np.median(valid_rep) / 100),
]
x = np.arange(len(categories))
w = 0.35
ax_c.bar(x - w/2, disc_vals, w, label='Discovery', color='#3498db', alpha=0.8)
ax_c.bar(x + w/2, rep_vals,  w, label='Replicate',  color='#9b59b6', alpha=0.8)
ax_c.set_xticks(x)
ax_c.set_xticklabels(categories, fontsize=10)
ax_c.set_ylabel('Median Bootstrap CI Width', fontsize=11, fontweight='bold')
ax_c.set_title('C) CI Width: M1 vs M2 Real vs M2 Permutation Median',
               fontsize=11, fontweight='bold')
ax_c.legend(fontsize=10)
ax_c.grid(axis='y', alpha=0.3)

# Panel D: Summary table
ax_d = fig.add_subplot(gs[1, 1])
ax_d.axis('off')
table_data = [
    ['', 'Discovery', 'Replicate'],
    ['Real M2 reduction (median)',  f"{real_reduction_disc:.1f}%",                   f"{real_reduction_rep:.1f}%"],
    ['Perm mean reduction',         f"{valid_disc.mean():.1f}%",                     f"{valid_rep.mean():.1f}%"],
    ['Perm 95th pctile',            f"{np.percentile(valid_disc, 95):.1f}%",         f"{np.percentile(valid_rep, 95):.1f}%"],
    ['One-sided p-value',           f"{p_val_disc:.4f}",                             f"{p_val_rep:.4f}"],
    ['Mol features used',           str(len(mol_sel_features)),                      str(len(mol_sel_features))],
    ['N permutations',              str(len(valid_disc)),                            str(len(valid_rep))],
]
tbl = ax_d.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc='center', loc='center', colWidths=[0.45, 0.25, 0.25])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.2)
for j in range(3):
    tbl[(0, j)].set_facecolor('#2c3e50')
    tbl[(0, j)].set_text_props(color='white', weight='bold')
for i in range(1, len(table_data)):
    if i % 2 == 0:
        for j in range(3):
            tbl[(i, j)].set_facecolor('#ecf0f1')
ax_d.set_title('D) Permutation Test Summary', fontsize=11, fontweight='bold', pad=15)

fig.suptitle(
    'Permutation Sanity Check: Is the M2 CI Width Reduction Driven by Molecular Biology?\n'
    f'Real M2 baseline: saved model risk scores (identical to reported results) | '
    f'Permuted: plain CoxPH on {len(mol_sel_features)} selected features, shuffled outcomes | '
    f'{N_PERMUTATIONS} × {N_BOOTSTRAP} bootstrap iterations | Reduction metric: median CI width',
    fontsize=10, fontweight='bold', y=1.01
)

plt.savefig(os.path.join(OUTPUT_DIR, 'FIGURE_Permutation_Sanity_Check.jpg'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'FIGURE_Permutation_Sanity_Check.pdf'),
            bbox_inches='tight')
plt.close()
print("  Figure saved: FIGURE_Permutation_Sanity_Check.jpg/pdf")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("PERMUTATION SANITY CHECK COMPLETE")
print("=" * 80)
print(f"\n  Real M2 risk scores:   loaded from saved CSVs (same as uncertainty script)")
print(f"  Permuted risk scores:  plain CoxPH on {len(mol_sel_features)} selected features, shuffled outcomes")
print(f"  Reduction metric:      median CI width (consistent with publication table)")
print(f"  Selected features:     {mol_sel_features}")
print(f"\n  N permutations run:    {N_PERMUTATIONS} (successful: {len(valid_disc)})")
print(f"  N bootstrap per perm:  {N_BOOTSTRAP}")
print(f"\n  Discovery:")
print(f"    M1 median CI width:  {np.median(m1_ci_disc):.4f}")
print(f"    M2 median CI width:  {np.median(m2_ci_disc_real):.4f}")
print(f"    Real reduction:      {real_reduction_disc:.1f}%")
print(f"    Permutation mean:    {valid_disc.mean():.1f}%  (95th: {np.percentile(valid_disc, 95):.1f}%)")
print(f"    One-sided p-value:   {p_val_disc:.4f}")
print(f"\n  Replicate:")
print(f"    M1 median CI width:  {np.median(m1_ci_rep):.4f}")
print(f"    M2 median CI width:  {np.median(m2_ci_rep_real):.4f}")
print(f"    Real reduction:      {real_reduction_rep:.1f}%")
print(f"    Permutation mean:    {valid_rep.mean():.1f}%  (95th: {np.percentile(valid_rep, 95):.1f}%)")
print(f"    One-sided p-value:   {p_val_rep:.4f}")
print(f"\n  Results saved to: {OUTPUT_DIR}")
print("=" * 80)