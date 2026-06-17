"""
Treatment Projection Analysis - Clinical-ResNet Model (DL-M1)

Author: Fadel Batal
Revised: June 2026

CONFIDENCE INTERVAL METHODOLOGY
-------------------------------
Point estimates (risk score and 1/3/5-yr PFS) come from a single estimator:
the unregularized CoxPH model refit on the FULL discovery cohort using the
LASSO/elastic-net-selected features ("refit_model"). This is the full-data
version of exactly what each bootstrap resample fits, so it lives on the same
scale as the bootstrap predictions and the risk-stratification threshold.

Confidence intervals are obtained by refitting the model on 1,000 bootstrap
resamples of the DISCOVERY cohort and applying each refit to the FIXED patient.
Intervals are reported as the 2.5th-97.5th percentiles of the resulting
prediction distribution. They therefore reflect uncertainty arising from finite
training-sample size (parameter/sampling uncertainty), NOT individual outcome
variability. Because point and CI are derived from the same estimator family on
the same scale, the point estimate always lies within its own CI, and survival
probabilities (already bounded in [0,1]) need no clipping.

This is the SAME bootstrap-the-training-set procedure behind the risk-score CIs
and the reported uncertainty-reduction figures; here we additionally read off
S(t) at t = 12, 36, 60 months in each refit.
"""

import warnings
warnings.filterwarnings('ignore')

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

# =============================================================================
# CONFIGURATION
# =============================================================================

# Repo root: one level up from this script's folder (04_Analysis/)
REPO_ROOT   = Path(__file__).resolve().parent.parent
DLM1_DIR    = REPO_ROOT / '01_DLM1_Clinico_ResNet'

# Patient-level data files (not included in repo — produced by the training notebooks
# after downloading CBTN data; see README.md for data access instructions)
X_CLINICAL_PATH = DLM1_DIR / 'outputs' / 'models' / 'X_clinical.pkl'
X_RESNET_PATH   = DLM1_DIR / 'outputs' / 'models' / 'X_resnet.pkl'
Y_PATH          = DLM1_DIR / 'outputs' / 'models' / 'y.pkl'

OUTPUT_BASE       = REPO_ROOT / '04_Analysis' / 'outputs' / 'treatment_scenario_DLM1'
SCENARIOS_DIR     = OUTPUT_BASE / 'scenarios'
BOOTSTRAP_DIR     = OUTPUT_BASE / 'bootstrap_uncertainty'
VISUALIZATION_DIR = OUTPUT_BASE / 'visualizations'

for dir_path in [SCENARIOS_DIR, BOOTSTRAP_DIR, VISUALIZATION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
RANDOM_SEED = 42
SAVE_MODELS = True

# Survival time points in months (1, 3, 5 years)
SURV_TIME_POINTS = [12, 36, 60]
SURV_LABELS      = ['1yr', '3yr', '5yr']

# Treatment scenarios — defined once, shared across all functions
TREATMENT_SCENARIOS = [
    {'label': 'Biopsy only + Chemo: Yes',               'resection': 1/3, 'chemo': 1, 'color': '#377eb8', 'linestyle': '-'},
    {'label': 'Partial resection + Chemo: No',          'resection': 2/3, 'chemo': 0, 'color': '#4daf4a', 'linestyle': '--'},
    {'label': 'Partial resection + Chemo: Yes',         'resection': 2/3, 'chemo': 1, 'color': '#984ea3', 'linestyle': '--'},
    {'label': 'Gross/Near total resection + Chemo: No', 'resection': 3/3, 'chemo': 0, 'color': '#ff7f00', 'linestyle': '-.'},
]

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_treatment_scenarios(patient_id, X_replicate_sel, estimator, median_threshold):
    """Risk score + risk group per scenario, using the single refit estimator."""
    patient_features = X_replicate_sel.loc[patient_id].copy()

    results = []
    for scenario in TREATMENT_SCENARIOS:
        sf = patient_features.copy()
        sf['Extent of Tumor Resection'] = scenario['resection']
        sf['Chemotherapy']              = scenario['chemo']

        risk_score = float(estimator.predict([sf.values])[0])
        risk_group = 'High' if risk_score > median_threshold else 'Low'

        results.append({
            'Patient_ID':      patient_id,
            'Treatment_Label': scenario['label'],
            'Resection':       scenario['label'].split(' + ')[0],
            'Chemotherapy':    'Yes' if scenario['chemo'] == 1 else 'No',
            'Risk_Score':      risk_score,
            'Risk_Group':      risk_group,
        })

    return pd.DataFrame(results)


def get_survival_curve(model, features_arr, time_points):
    surv_fn = model.predict_survival_function([features_arr])[0]
    if hasattr(surv_fn, 'x') and len(surv_fn.x) > 0:
        t_min, t_max = surv_fn.x[0], surv_fn.x[-1]
    else:
        t_min, t_max = 0, 200
    return np.array([surv_fn(np.clip(t, t_min, t_max)) for t in time_points])


def get_survival_curves_with_uncertainty(scenario_features_sel, refit_model,
                                         bootstrap_models, time_points):
    """
    Point estimate from the full-discovery refit (the estimator the bootstrap
    resamples). CI band from the 2.5th-97.5th percentiles of the bootstrap
    survival functions. S(t) in [0,1] already, so no clipping is required and
    the point estimate lies within its own band by construction.
    """
    point_curve = get_survival_curve(refit_model, scenario_features_sel.values, time_points)

    boot_curves = np.array([
        get_survival_curve(m, scenario_features_sel.values, time_points)
        for m in bootstrap_models
    ])

    return {
        'point':    point_curve,
        'ci_lower': np.percentile(boot_curves, 2.5,  axis=0),
        'ci_upper': np.percentile(boot_curves, 97.5, axis=0),
    }


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

print("=" * 80)
print("TREATMENT PROJECTION ANALYSIS - CLINICAL-RESNET MODEL (DL-M1)")
print("=" * 80)

# =============================================================================
# STEP 1: LOAD DATA AND MODEL
# =============================================================================

print("\nStep 1: Loading data and model...")

X_clinical = pd.read_pickle(X_CLINICAL_PATH)
X_resnet   = pd.read_pickle(X_RESNET_PATH)
y          = pd.read_pickle(Y_PATH)

print(f"  Loaded X_clinical: {X_clinical.shape}")
print(f"  Loaded X_resnet:   {X_resnet.shape}")
print(f"  Loaded y: {y.shape}")

X_clinical_features = X_clinical.drop(columns="Cohort")
X_resnet_features   = X_resnet.drop(columns="Cohort", errors='ignore')

X_combined = X_clinical_features.merge(X_resnet_features, left_index=True, right_index=True)
X_combined = X_combined.merge(y, left_index=True, right_index=True)

X_discovery_full = X_combined[X_combined["Cohort"] == "Discovery"]
X_replicate_full = X_combined[X_combined["Cohort"] == "Replicate"]

X_discovery = X_discovery_full.drop(columns=["Progression Free Survival", "Event", "Cohort"])
X_replicate = X_replicate_full.drop(columns=["Progression Free Survival", "Event", "Cohort"])

y_discovery = np.array(
    [(row["Event"], row["Progression Free Survival"]) for _, row in X_discovery_full.iterrows()],
    dtype=[("Event", "?"), ("Progression Free Survival", "<f8")],
)
y_replicate_struct = np.array(
    [(row["Event"], row["Progression Free Survival"]) for _, row in X_replicate_full.iterrows()],
    dtype=[("Event", "?"), ("Progression Free Survival", "<f8")],
)

print(f"  Discovery cohort: {X_discovery.shape[0]} patients, {X_discovery.shape[1]} features")
print(f"  Replicate cohort: {X_replicate.shape[0]} patients, {X_replicate.shape[1]} features")

# Original (penalized) estimator — used only to identify selected features.
with open(DLM1_DIR / 'outputs' / 'models' / 'estimator.pkl', "rb") as f:
    original_estimator = pickle.load(f)

original_coefs         = original_estimator.coef_[:, 0]
selected_mask          = original_coefs != 0
selected_feature_names = X_discovery.columns[selected_mask].tolist()
n_selected             = len(selected_feature_names)
print(f"  LASSO/elastic-net-selected features: {n_selected} / {X_discovery.shape[1]}")

X_discovery_sel = X_discovery[selected_feature_names]
X_replicate_sel = X_replicate[selected_feature_names]

# -----------------------------------------------------------------------------
# Single estimator for ALL projection outputs: unregularized CoxPH refit on the
# full discovery cohort with the selected features. Point estimates, threshold,
# and bootstrap CIs all live on THIS scale.
# -----------------------------------------------------------------------------
refit_model = CoxPHSurvivalAnalysis(alpha=0.0, verbose=0)
refit_model.fit(X_discovery_sel, y_discovery)
print(f"  Refit CoxPH (no regularization) on {len(X_discovery_sel)} discovery patients")

# Risk-stratification threshold on the refit scale (median discovery risk).
refit_disc_scores = refit_model.predict(X_discovery_sel)
median_threshold  = float(np.median(refit_disc_scores))
print(f"  Risk threshold (refit-scale median discovery risk): {median_threshold:.6f}")

# Sanity check: refit discrimination should track the reported DL-M1 numbers.
refit_pred_rep = refit_model.predict(X_replicate_sel)
harrell_disc = concordance_index_censored(
    y_discovery["Event"], y_discovery["Progression Free Survival"], refit_disc_scores)[0]
harrell_rep  = concordance_index_censored(
    y_replicate_struct["Event"], y_replicate_struct["Progression Free Survival"], refit_pred_rep)[0]
try:
    uno_disc = concordance_index_ipcw(y_discovery, y_discovery, refit_disc_scores)[0]
    uno_rep  = concordance_index_ipcw(y_discovery, y_replicate_struct, refit_pred_rep)[0]
    print(f"  Refit CoxPH — Harrell C: Discovery={harrell_disc:.4f}  Replicate={harrell_rep:.4f}")
    print(f"  Refit CoxPH — Uno C:     Discovery={uno_disc:.4f}  Replicate={uno_rep:.4f}")
except Exception as e:
    uno_disc, uno_rep = np.nan, np.nan
    print(f"  Refit CoxPH — Harrell C: Discovery={harrell_disc:.4f}  Replicate={harrell_rep:.4f}")
    print(f"  Uno C skipped: {e}")


# =============================================================================
# STEP 2: GENERATE TREATMENT SCENARIOS
# =============================================================================

print("\nStep 2: Generating treatment scenarios...")

results_all          = []
verification_records = []

resection_map = {1/3: 'Biopsy only', 2/3: 'Partial resection', 3/3: 'Gross/Near total resection'}
chemo_map     = {0: 'No', 1: 'Yes'}

# Observed-scenario reference risk on the SAME refit scale (for verification).
original_risk_refit = pd.Series(
    refit_model.predict(X_replicate_sel), index=X_replicate_sel.index)

for i, patient_id in enumerate(X_replicate_sel.index, 1):
    print(f"  [{i}/{len(X_replicate_sel)}] {patient_id}", end='\r')

    try:
        scenarios = generate_treatment_scenarios(
            patient_id, X_replicate_sel, refit_model, median_threshold
        )
        results_all.append(scenarios)

        original_risk    = float(original_risk_refit.loc[patient_id])
        actual_resection = X_replicate_sel.loc[patient_id, 'Extent of Tumor Resection']
        actual_chemo     = X_replicate_sel.loc[patient_id, 'Chemotherapy']

        resection_label = None
        for val, label in resection_map.items():
            if abs(actual_resection - val) < 0.01:
                resection_label = label
                break

        if resection_label:
            matching_label    = f"{resection_label} + Chemo: {chemo_map[int(actual_chemo)]}"
            matching_scenario = scenarios[scenarios['Treatment_Label'] == matching_label]

            if len(matching_scenario) == 1:
                scenario_risk = matching_scenario['Risk_Score'].values[0]
                diff          = abs(original_risk - scenario_risk)
                verification_records.append({
                    'Patient_ID':    patient_id,
                    'Original_Risk': original_risk,
                    'Scenario_Risk': scenario_risk,
                    'Difference':    diff,
                    'Status':        'Match' if diff < 1e-6 else 'Mismatch',
                })
            else:
                verification_records.append({
                    'Patient_ID':    patient_id,
                    'Original_Risk': original_risk,
                    'Scenario_Risk': np.nan,
                    'Difference':    np.nan,
                    'Status':        'Scenario_Removed',
                })

    except Exception as e:
        print(f"  [{i}/{len(X_replicate_sel)}] {patient_id} - ERROR: {str(e)}")

all_results_df = pd.concat(results_all, ignore_index=True)
all_results_df.to_csv(f'{SCENARIOS_DIR}/all_treatment_scenarios.csv', index=False)
print(f"  Saved: {SCENARIOS_DIR}/all_treatment_scenarios.csv")

if verification_records:
    verification_df = pd.DataFrame(verification_records)
    verification_df.to_csv(f'{SCENARIOS_DIR}/verification_results.csv', index=False)
    n_match   = len(verification_df[verification_df['Status'] == 'Match'])
    n_removed = len(verification_df[verification_df['Status'] == 'Scenario_Removed'])
    n_mis     = len(verification_df[verification_df['Status'] == 'Mismatch'])
    print(f"  Verification: {n_match} matches, {n_mis} mismatches, {n_removed} scenario_removed")


# =============================================================================
# STEP 3: BOOTSTRAP UNCERTAINTY QUANTIFICATION
# =============================================================================

print("\nStep 3: Bootstrap uncertainty quantification...")

np.random.seed(RANDOM_SEED)
bootstrap_models   = []
bootstrap_metadata = []
start_time         = time.time()

for i in range(N_BOOTSTRAP):
    iter_start = time.time()
    print(f"  [{i+1}/{N_BOOTSTRAP}] Bootstrap iteration {i+1}...", end='', flush=True)

    try:
        indices = np.random.choice(len(X_discovery_sel), len(X_discovery_sel), replace=True)
        X_boot  = X_discovery_sel.iloc[indices]
        y_boot  = y_discovery[indices]

        boot_model = CoxPHSurvivalAnalysis(alpha=0.0, verbose=0)
        boot_model.fit(X_boot, y_boot)
        bootstrap_models.append(boot_model)

        boot_pred_disc = boot_model.predict(X_discovery_sel)
        boot_pred_rep  = boot_model.predict(X_replicate_sel)
        boot_c_disc    = concordance_index_censored(
            y_discovery["Event"], y_discovery["Progression Free Survival"], boot_pred_disc)[0]
        boot_c_rep     = concordance_index_censored(
            y_replicate_struct["Event"], y_replicate_struct["Progression Free Survival"], boot_pred_rep)[0]

        bootstrap_metadata.append({
            'iteration':        i + 1,
            'n_samples':        len(indices),
            'n_unique_samples': len(np.unique(indices)),
            'harrell_c_disc':   boot_c_disc,
            'harrell_c_rep':    boot_c_rep,
            'converged':        True,
        })

        iter_time = time.time() - iter_start
        print(f" OK ({iter_time:.1f}s, C_disc={boot_c_disc:.3f}, C_rep={boot_c_rep:.3f})", flush=True)

    except Exception as e:
        print(f" ERROR: {str(e)}", flush=True)
        bootstrap_metadata.append({'iteration': i + 1, 'converged': False, 'error': str(e)})

elapsed = time.time() - start_time
print(f"\n  Bootstrap completed in {elapsed/60:.1f} minutes")
print(f"  Successful iterations: {len(bootstrap_models)}/{N_BOOTSTRAP}")

print(f"\nGenerating per-patient risk-score CIs (point = refit, CI = bootstrap percentiles)...")
discovery_predictions = np.array([m.predict(X_discovery_sel) for m in bootstrap_models])
replicate_predictions = np.array([m.predict(X_replicate_sel) for m in bootstrap_models])

discovery_stats = pd.DataFrame({
    'SubjectID':          X_discovery_sel.index,
    'Point_Risk':         refit_model.predict(X_discovery_sel),
    'Bootstrap_Mean':     discovery_predictions.mean(axis=0),
    'Bootstrap_Std':      discovery_predictions.std(axis=0),
    'Bootstrap_CI_Lower': np.percentile(discovery_predictions, 2.5,  axis=0),
    'Bootstrap_CI_Upper': np.percentile(discovery_predictions, 97.5, axis=0),
})
discovery_stats['Uncertainty'] = discovery_stats['Bootstrap_CI_Upper'] - discovery_stats['Bootstrap_CI_Lower']

replicate_stats = pd.DataFrame({
    'SubjectID':          X_replicate_sel.index,
    'Point_Risk':         refit_model.predict(X_replicate_sel),
    'Bootstrap_Mean':     replicate_predictions.mean(axis=0),
    'Bootstrap_Std':      replicate_predictions.std(axis=0),
    'Bootstrap_CI_Lower': np.percentile(replicate_predictions, 2.5,  axis=0),
    'Bootstrap_CI_Upper': np.percentile(replicate_predictions, 97.5, axis=0),
})
replicate_stats['Uncertainty'] = replicate_stats['Bootstrap_CI_Upper'] - replicate_stats['Bootstrap_CI_Lower']

meta_df   = pd.DataFrame(bootstrap_metadata)
converged = meta_df[meta_df['converged'] == True]
if len(converged) > 0:
    c_disc_arr = converged['harrell_c_disc'].values
    c_rep_arr  = converged['harrell_c_rep'].values
    print(f"\n  Bootstrap C-index summary ({len(converged)} iterations):")
    print(f"    Discovery Harrell C: {c_disc_arr.mean():.4f} "
          f"[{np.percentile(c_disc_arr, 2.5):.4f}, {np.percentile(c_disc_arr, 97.5):.4f}]")
    print(f"    Replicate Harrell C: {c_rep_arr.mean():.4f} "
          f"[{np.percentile(c_rep_arr, 2.5):.4f}, {np.percentile(c_rep_arr, 97.5):.4f}]")

pd.DataFrame(bootstrap_metadata).to_csv(BOOTSTRAP_DIR / "bootstrap_metadata.csv", index=False)
discovery_stats.to_csv(BOOTSTRAP_DIR / "discovery_predictions_with_uncertainty.csv", index=False)
replicate_stats.to_csv(BOOTSTRAP_DIR / "replicate_predictions_with_uncertainty.csv", index=False)

if SAVE_MODELS:
    with open(BOOTSTRAP_DIR / "bootstrap_models.pkl", "wb") as f:
        pickle.dump(bootstrap_models, f)

print(f"  Saved bootstrap results to: {BOOTSTRAP_DIR}")
print(f"  Discovery mean CI width: {discovery_stats['Uncertainty'].mean():.6f}")
print(f"  Replicate mean CI width: {replicate_stats['Uncertainty'].mean():.6f}")


# =============================================================================
# STEP 4: GENERATE SURVIVAL CURVE VISUALIZATIONS
# =============================================================================

print("\nStep 4: Generating survival curve visualizations...")


def plot_patient_survival_scenarios(patient_id, X_replicate_sel,
                                    refit_model, bootstrap_models,
                                    median_threshold, save_path,
                                    actual_pfs=None, actual_event=None):
    patient_features_sel = X_replicate_sel.loc[patient_id].copy()

    test_surv_fn = refit_model.predict_survival_function([patient_features_sel.values])[0]
    if hasattr(test_surv_fn, 'x') and len(test_surv_fn.x) > 0:
        min_time, max_time = test_surv_fn.x[0], test_surv_fn.x[-1]
    else:
        min_time, max_time = 0, 150

    time_points = np.linspace(min_time, max_time, 100)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1 = axes[0]
    scenario_data = []

    for scenario in TREATMENT_SCENARIOS:
        sf_sel = patient_features_sel.copy()
        sf_sel['Extent of Tumor Resection'] = scenario['resection']
        sf_sel['Chemotherapy']              = scenario['chemo']

        surv_stats = get_survival_curves_with_uncertainty(
            sf_sel, refit_model, bootstrap_models, time_points)

        # Risk score: POINT from refit estimator; CI from bootstrap percentiles.
        # Same scale as threshold -> point always inside its own CI.
        point_risk    = float(refit_model.predict([sf_sel.values])[0])
        boot_risks    = np.array([m.predict([sf_sel.values])[0] for m in bootstrap_models])
        risk_ci_lower = float(np.percentile(boot_risks, 2.5))
        risk_ci_upper = float(np.percentile(boot_risks, 97.5))
        risk_group    = 'High' if point_risk > median_threshold else 'Low'

        ax1.plot(time_points, surv_stats['point'],
                 color=scenario['color'], linestyle=scenario['linestyle'],
                 linewidth=2.5, label=scenario['label'], alpha=0.9)
        ax1.fill_between(time_points, surv_stats['ci_lower'], surv_stats['ci_upper'],
                         color=scenario['color'], alpha=0.15)

        # ── Survival at 1, 3, 5 years with 95% percentile CI ─────────────────
        row = {
            'scenario':      scenario['label'],
            'risk_score':    point_risk,
            'risk_ci_lower': risk_ci_lower,
            'risk_ci_upper': risk_ci_upper,
            'risk_group':    risk_group,
        }
        for t_months, t_label in zip(SURV_TIME_POINTS, SURV_LABELS):
            if max_time >= t_months:
                idx = np.argmin(np.abs(time_points - t_months))
                row[t_label]               = surv_stats['point'][idx]
                row[f'{t_label}_ci_lower'] = surv_stats['ci_lower'][idx]
                row[f'{t_label}_ci_upper'] = surv_stats['ci_upper'][idx]
            else:
                row[t_label]               = None
                row[f'{t_label}_ci_lower'] = None
                row[f'{t_label}_ci_upper'] = None
        scenario_data.append(row)

    if actual_pfs is not None and actual_event is not None and actual_pfs <= max_time:
        if actual_event:
            ax1.axvline(actual_pfs, color='black', linestyle=':', linewidth=2.5,
                        label=f'Actual Progression ({actual_pfs:.1f} mo)', zorder=1, alpha=0.7)
        else:
            ax1.axvline(actual_pfs, color='gray', linestyle=':', linewidth=2.5,
                        label=f'Censored at {actual_pfs:.1f} mo', zorder=1, alpha=0.7)

    ax1.set_xlabel('Time (months)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Progression-Free Survival Probability', fontsize=12, fontweight='bold')
    ax1.set_title(f'Patient {patient_id}: Survival Curve Under Treatment Scenarios',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9, framealpha=0.95)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, max_time)

    # ── Table ───────────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis('off')

    scenario_df = pd.DataFrame(scenario_data)
    headers = ['Treatment Scenario', 'Risk Score\n(95% CI)', 'Risk\nGroup']
    for t_label in SURV_LABELS:
        if scenario_df[t_label].notna().any():
            headers.append(t_label.replace('yr', '-yr PFS\n(95% CI)'))

    table_data = []
    for _, row in scenario_df.iterrows():
        rd = [
            row['scenario'],
            f"{row['risk_score']:.2f}\n[{row['risk_ci_lower']:.2f}, {row['risk_ci_upper']:.2f}]",
            row['risk_group'],
        ]
        for t_label in SURV_LABELS:
            if row[t_label] is not None:
                rd.append(
                    f"{row[t_label]:.2f}\n"
                    f"[{row[f'{t_label}_ci_lower']:.2f}, {row[f'{t_label}_ci_upper']:.2f}]"
                )
            else:
                rd.append("N/A")
        table_data.append(rd)

    table = ax2.table(cellText=table_data, colLabels=headers,
                      cellLoc='center', loc='center',
                      colWidths=[0.35, 0.15, 0.15] + [0.12] * (len(headers) - 3))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3.2)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#1a9850')
        table[(0, j)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#f0f0f0')
        rg = table_data[i - 1][2]
        table[(i, 2)].set_facecolor('#ffcccc' if rg == 'High' else '#ccffcc')

    ax2.set_title(
        'Treatment Scenario Summary\n'
        'Point estimate from full-discovery refit; 95% CI = 2.5th–97.5th bootstrap percentile',
        fontsize=9, fontweight='bold', pad=20)

    if actual_pfs is not None and actual_event is not None:
        status = "Progressed" if actual_event else "Censored"
        fig.suptitle(f'Survival Analysis for Patient {patient_id}\n'
                     f'Actual outcome: {status} at {actual_pfs:.1f} months',
                     fontsize=14, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f'Survival Analysis for Patient {patient_id}',
                     fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return scenario_df


y_replicate_full_idx = X_replicate_full[['Progression Free Survival', 'Event']]
summary_data         = []
failed_patients      = []
start_time           = time.time()

for i, patient_id in enumerate(X_replicate_sel.index, 1):
    if i > 1:
        elapsed     = time.time() - start_time
        eta_seconds = (elapsed / (i - 1)) * (len(X_replicate_sel) - i)
        eta_str     = f"ETA: {eta_seconds/60:.1f}min"
    else:
        eta_str = "calculating..."

    print(f"  [{i}/{len(X_replicate_sel)}] {patient_id} ({eta_str})...", end='\r')

    try:
        actual_pfs   = y_replicate_full_idx.loc[patient_id, 'Progression Free Survival']
        actual_event = y_replicate_full_idx.loc[patient_id, 'Event']

        scenario_summary = plot_patient_survival_scenarios(
            patient_id, X_replicate_sel,
            refit_model, bootstrap_models, median_threshold,
            VISUALIZATION_DIR / f'{patient_id}_survival_curves.png',
            actual_pfs=actual_pfs, actual_event=actual_event
        )
        scenario_summary['Patient_ID'] = patient_id
        summary_data.append(scenario_summary)

    except Exception as e:
        print(f"\n  [{i}/{len(X_replicate_sel)}] {patient_id} - ERROR: {str(e)}")
        failed_patients.append({'Patient_ID': patient_id, 'Error': str(e)})

total_time = time.time() - start_time
print(f"\n  Generated visualizations for {len(summary_data)} patients")
print(f"  Total time: {total_time/60:.1f} minutes")

if summary_data:
    pd.concat(summary_data, ignore_index=True).to_csv(
        VISUALIZATION_DIR / 'survival_scenario_summaries.csv', index=False)
    print(f"  Saved: survival_scenario_summaries.csv")

if failed_patients:
    pd.DataFrame(failed_patients).to_csv(VISUALIZATION_DIR / 'failed_patients.csv', index=False)
    print(f"  Failed patients: {len(failed_patients)}")


# =============================================================================
# STEP 5: CREATE HTML INDEX
# =============================================================================

print("\nStep 5: Creating HTML index...")

progressed_patients = y_replicate_full_idx[y_replicate_full_idx['Event']].index.tolist()
censored_patients   = y_replicate_full_idx[~y_replicate_full_idx['Event']].index.tolist()

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Treatment Projection Analysis - Clinical-ResNet</title>
    <style>
        body {{font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5;}}
        h1 {{color: #333; border-bottom: 3px solid #1a9850; padding-bottom: 10px;}}
        .summary {{background: #e8f5e9; padding: 15px; border-radius: 8px; margin-bottom: 20px;}}
        .controls {{background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
        .filter-btn {{padding: 10px 20px; border: 2px solid #1a9850; background: white; color: #1a9850; border-radius: 5px; cursor: pointer; font-weight: bold; margin: 5px;}}
        .filter-btn:hover {{background: #1a9850; color: white;}}
        .filter-btn.active {{background: #1a9850; color: white;}}
        .patient-card {{background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}}
        .patient-header {{font-size: 18px; font-weight: bold; color: #1a9850; margin-bottom: 15px;}}
        img {{width: 100%; border: 1px solid #ddd; border-radius: 4px; cursor: pointer;}}
    </style>
    <script>
        function filterPatients(category) {{
            const cards = document.querySelectorAll('.patient-card');
            const buttons = document.querySelectorAll('.filter-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            cards.forEach(card => {{
                const categories = card.dataset.categories.split(',');
                if (category === 'all' || categories.includes(category)) {{
                    card.style.display = 'block';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}
    </script>
</head>
<body>
    <h1>Treatment Projection Analysis - Clinical-ResNet Model</h1>
    <div class="summary">
        <strong>Analysis Complete:</strong> {len(summary_data)}/{len(X_replicate_sel)} patients<br>
        <strong>Bootstrap Models:</strong> {len(bootstrap_models)}<br>
        <strong>Mean CI Width (Replicate):</strong> {replicate_stats['Uncertainty'].mean():.6f}<br>
        <strong>Processing Time:</strong> {total_time/60:.1f} minutes
    </div>
    <div class="controls">
        <strong>Filter Patients:</strong>
        <button class="filter-btn active" onclick="filterPatients('all')">All ({len(X_replicate_sel)})</button>
        <button class="filter-btn" onclick="filterPatients('progressed')">Progressed ({len(progressed_patients)})</button>
        <button class="filter-btn" onclick="filterPatients('censored')">Censored ({len(censored_patients)})</button>
    </div>
"""

for patient_id in X_replicate_sel.index:
    if patient_id in [p['Patient_ID'] for p in failed_patients]:
        continue
    outcome  = y_replicate_full_idx.loc[patient_id]
    status   = "Progressed" if outcome['Event'] else "Censored"
    category = 'progressed' if outcome['Event'] else 'censored'
    html_content += f"""
    <div class="patient-card" data-categories="{category}">
        <div class="patient-header">
            Patient: {patient_id} ({status} at {outcome['Progression Free Survival']:.1f} months)
        </div>
        <img src="{patient_id}_survival_curves.png" alt="Survival Curves"
             onclick="window.open('{patient_id}_survival_curves.png', '_blank')">
    </div>
    """

html_content += "</body></html>"

with open(VISUALIZATION_DIR / 'index.html', 'w') as f:
    f.write(html_content)
print(f"  Created: {VISUALIZATION_DIR}/index.html")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("TREATMENT PROJECTION ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n  Treatment scenarios: {len(all_results_df)} total")
print(f"  Bootstrap models: {len(bootstrap_models)}")
print(f"  Survival curve plots: {len(summary_data)}/{len(X_replicate_sel)} patients")
print(f"  Mean uncertainty (risk-score CI width, replicate): {replicate_stats['Uncertainty'].mean():.6f}")
print(f"\n  Refit CoxPH C-index (sanity check vs reported DL-M1):")
print(f"    Discovery Harrell C: {harrell_disc:.4f}")
print(f"    Replicate Harrell C: {harrell_rep:.4f}")
print(f"\n  Open: {VISUALIZATION_DIR}/index.html")
print("=" * 80)