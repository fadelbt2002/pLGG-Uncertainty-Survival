"""
Statistical Comparison of Survival Models: Clinical-Radiomic vs Clinical-ResNet
Bootstrap C-index Comparison (Harrell's C + Uno's C) with Coverage Validation

Methodology:
  - Paired bootstrap: same resampled indices applied to both models in each iteration,
    ensuring the comparison is internally consistent and the difference distribution
    captures correlated model uncertainty rather than independent noise.
  - Discovery cohort: bootstrap resamples from discovery patients only.
  - Replicate cohort: bootstrap resamples from replicate patients only (held-out;
    no leakage from discovery).
  - Two-sided p-value derived from the bootstrap null distribution of Δ C-index.
  - Uno's C computed with tau = 90th percentile of discovery event times, consistent
    with the tau convention used throughout the survival modeling pipeline.
  - Coverage validation reports how many subjects were sampled at least once across
    all 1000 bootstrap iterations (expected: ~63.2% per iteration on average).

Author: Fadel Batal
Institution: Center for Data-Driven Discovery in Biomedicine, CHOP
Date: 2026-02-05
"""

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Publication-quality settings
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial']
})

# =============================================================================
# CONFIGURATION
# =============================================================================

# Repo root: one level up from this script's folder (04_Analysis/)
REPO_ROOT    = Path(__file__).resolve().parent.parent
DLM1_DIR     = REPO_ROOT / '01_DLM1_Clinico_ResNet'

# The radiomic model is NOT included in this repository (it is the baseline
# comparison model). Point RADIOMIC_DIR to its outputs/data/ folder if you
# have run the Clinico-Radiomic_Model pipeline separately.
RADIOMIC_DIR = Path(os.environ.get('RADIOMIC_DIR', str(REPO_ROOT / 'Clinico-Radiomic_Model' / 'outputs')))
RESNET_DIR   = DLM1_DIR / 'outputs'
N_BOOTSTRAP  = 1000
RANDOM_SEED  = 42

timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = str(REPO_ROOT / '04_Analysis' / 'outputs' / f'statistical_comparison_{timestamp}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("BOOTSTRAP C-INDEX COMPARISON: CLINICAL-RADIOMIC VS CLINICAL-RESNET")
print("="*80)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_data(rad_df, res_df, cohort_name):
    """
    Align radiomic and resnet dataframes on common SubjectIDs.
    Outcomes taken from radiomic df (identical to resnet df for the same patients).
    """
    common_ids = sorted(list(set(rad_df.index) & set(res_df.index)))
    rad_df = rad_df.loc[common_ids]
    res_df = res_df.loc[common_ids]
    return {
        'event':     rad_df['Event'].values.astype(bool),
        'time':      rad_df['Progression Free Survival'].values,
        'rad_risk':  rad_df['Risk Score'].values,
        'res_risk':  res_df['Risk Score'].values,
        'n':         len(common_ids),
        'n_events':  int(np.sum(rad_df['Event'].values.astype(bool))),
        'cohort':    cohort_name,
        'ids':       common_ids,
    }


def make_structured(event, time):
    return np.array(
        [(bool(e), float(t)) for e, t in zip(event, time)],
        dtype=[('Event', bool), ('Progression Free Survival', float)]
    )


def bootstrap_c_index_comparison(event, time, risk1, risk2,
                                  disc_event=None, disc_time=None,
                                  n_bootstrap=1000, seed=42,
                                  cohort_label=''):
    """
    Paired bootstrap comparison of Harrell's C and Uno's C for two models.

    Parameters
    ----------
    event, time, risk1, risk2 : arrays for the cohort being evaluated
    disc_event, disc_time     : discovery outcomes — required for Uno's C IPCW
                                weights (set to same as event/time for discovery,
                                set to discovery arrays for replicate).
    cohort_label              : 'Discovery' or 'Replicate' (for printed output)

    Coverage statistic: counts how many times each subject appears across all
    bootstrap iterations combined (not per-iteration). Expected mean ≈ 0.632 × n_bootstrap.
    """
    np.random.seed(seed)
    print(f"    Running {n_bootstrap} bootstrap iterations...")

    n_samples = len(event)
    y_full    = make_structured(event, time)

    # Uno tau: 90th percentile of discovery event times (pipeline convention)
    disc_event_times = disc_time[disc_event]
    tau = float(np.percentile(disc_event_times, 90)) if len(disc_event_times) > 0 else None
    y_disc_full = make_structured(disc_event, disc_time)

    # Storage
    harrell1_boots, harrell2_boots = [], []
    uno1_boots,     uno2_boots     = [], []
    harrell_diff_boots             = []
    uno_diff_boots                 = []

    # Coverage tracking (cumulative across all iterations)
    subject_sample_count = np.zeros(n_samples, dtype=int)

    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            print(f"      Progress: {i+1}/{n_bootstrap}")

        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Coverage: count how many times each subject is selected cumulatively
        for idx in np.unique(indices):
            subject_sample_count[idx] += 1

        try:
            e_b    = event[indices]
            t_b    = time[indices]
            r1_b   = risk1[indices]
            r2_b   = risk2[indices]
            y_b    = make_structured(e_b, t_b)

            # Harrell's C (paired, same bootstrap sample)
            h1 = concordance_index_censored(e_b, t_b, r1_b)[0]
            h2 = concordance_index_censored(e_b, t_b, r2_b)[0]
            harrell1_boots.append(h1)
            harrell2_boots.append(h2)
            harrell_diff_boots.append(h1 - h2)

            # Uno's C — IPCW weights from discovery (consistent with pipeline)
            if tau is not None:
                try:
                    u1 = concordance_index_ipcw(y_disc_full, y_b, r1_b, tau=tau)[0]
                    u2 = concordance_index_ipcw(y_disc_full, y_b, r2_b, tau=tau)[0]
                    uno1_boots.append(u1)
                    uno2_boots.append(u2)
                    uno_diff_boots.append(u1 - u2)
                except Exception:
                    pass

        except Exception:
            continue

    harrell1_boots     = np.array(harrell1_boots)
    harrell2_boots     = np.array(harrell2_boots)
    harrell_diff_boots = np.array(harrell_diff_boots)
    uno1_boots         = np.array(uno1_boots)
    uno2_boots         = np.array(uno2_boots)
    uno_diff_boots     = np.array(uno_diff_boots)

    # ── Coverage diagnostics ──────────────────────────────────────────────────
    subject_in_sample  = subject_sample_count > 0
    n_ever_sampled     = int(np.sum(subject_in_sample))
    n_never_sampled    = n_samples - n_ever_sampled
    mean_sample_count  = float(np.mean(subject_sample_count))
    min_sample_count   = int(np.min(subject_sample_count))
    max_sample_count   = int(np.max(subject_sample_count))
    # Expected mean per subject = n_bootstrap × (1 − e^{−1}) ≈ 632 for 1000 iterations
    expected_mean      = n_bootstrap * (1 - np.exp(-1))

    print(f"      Bootstrap coverage (cumulative across all {n_bootstrap} iterations):")
    print(f"        Subjects sampled at least once: {n_ever_sampled}/{n_samples} "
          f"({n_ever_sampled/n_samples*100:.1f}%)")
    print(f"        Mean appearances per subject: {mean_sample_count:.1f}  "
          f"(expected ~{expected_mean:.1f})  range=[{min_sample_count}, {max_sample_count}]")
    if n_never_sampled > 0:
        print(f"        WARNING: {n_never_sampled} subject(s) never sampled — "
              f"very small cohort or extreme outliers")

    # ── Point estimates on full cohort ────────────────────────────────────────
    harrell1_orig = concordance_index_censored(event, time, risk1)[0]
    harrell2_orig = concordance_index_censored(event, time, risk2)[0]
    harrell_diff_orig = harrell1_orig - harrell2_orig

    uno1_orig, uno2_orig, uno_diff_orig = np.nan, np.nan, np.nan
    if tau is not None:
        try:
            uno1_orig = concordance_index_ipcw(y_disc_full, y_full, risk1, tau=tau)[0]
            uno2_orig = concordance_index_ipcw(y_disc_full, y_full, risk2, tau=tau)[0]
            uno_diff_orig = uno1_orig - uno2_orig
        except Exception:
            pass

    # ── p-values (two-sided, bootstrap permutation style) ─────────────────────
    def two_sided_p(diff_arr):
        if len(diff_arr) == 0:
            return np.nan
        prop_neg = np.mean(diff_arr <= 0)
        return float(2 * min(prop_neg, 1 - prop_neg))

    def sig_label(p):
        # Conservative labels appropriate for small survival cohorts
        if np.isnan(p):  return 'N/A'
        if p < 0.001:    return 'p<0.001'
        if p < 0.01:     return 'p<0.01'
        if p < 0.05:     return 'p<0.05'
        return 'ns'

    def interpretation(p):
        # Deliberately neutral phrasing — CI is the primary inferential quantity
        if np.isnan(p):  return 'Uno C not available'
        if p < 0.05:     return 'Statistically significant difference'
        return 'No statistically significant difference'

    harrell_p = two_sided_p(harrell_diff_boots)
    uno_p     = two_sided_p(uno_diff_boots)

    def ci(arr, lo=2.5, hi=97.5):
        if len(arr) == 0:
            return np.nan, np.nan
        return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

    h_lo, h_hi   = ci(harrell_diff_boots)
    u_lo, u_hi   = ci(uno_diff_boots)
    h1_lo, h1_hi = ci(harrell1_boots)
    h2_lo, h2_hi = ci(harrell2_boots)
    u1_lo, u1_hi = ci(uno1_boots)
    u2_lo, u2_hi = ci(uno2_boots)

    return {
        # Harrell's C
        'harrell1_orig':    harrell1_orig,
        'harrell1_mean':    float(np.mean(harrell1_boots)),
        'harrell1_se':      float(np.std(harrell1_boots)),
        'harrell1_ci':      (h1_lo, h1_hi),
        'harrell1_boots':   harrell1_boots,
        'harrell2_orig':    harrell2_orig,
        'harrell2_mean':    float(np.mean(harrell2_boots)),
        'harrell2_se':      float(np.std(harrell2_boots)),
        'harrell2_ci':      (h2_lo, h2_hi),
        'harrell2_boots':   harrell2_boots,
        'harrell_diff_orig': harrell_diff_orig,
        'harrell_diff_mean': float(np.mean(harrell_diff_boots)),
        'harrell_diff_se':   float(np.std(harrell_diff_boots)),
        'harrell_diff_ci':   (h_lo, h_hi),
        'harrell_diff_boots': harrell_diff_boots,
        'harrell_p':         harrell_p,
        'harrell_sig':       sig_label(harrell_p),
        'harrell_interp':    interpretation(harrell_p),
        # Uno's C
        'uno1_orig':         uno1_orig,
        'uno1_mean':         float(np.mean(uno1_boots)) if len(uno1_boots) else np.nan,
        'uno1_se':           float(np.std(uno1_boots))  if len(uno1_boots) else np.nan,
        'uno1_ci':           (u1_lo, u1_hi),
        'uno1_boots':        uno1_boots,
        'uno2_orig':         uno2_orig,
        'uno2_mean':         float(np.mean(uno2_boots)) if len(uno2_boots) else np.nan,
        'uno2_se':           float(np.std(uno2_boots))  if len(uno2_boots) else np.nan,
        'uno2_ci':           (u2_lo, u2_hi),
        'uno2_boots':        uno2_boots,
        'uno_diff_orig':     uno_diff_orig,
        'uno_diff_mean':     float(np.mean(uno_diff_boots)) if len(uno_diff_boots) else np.nan,
        'uno_diff_se':       float(np.std(uno_diff_boots))  if len(uno_diff_boots) else np.nan,
        'uno_diff_ci':       (u_lo, u_hi),
        'uno_diff_boots':    uno_diff_boots,
        'uno_p':             uno_p,
        'uno_sig':           sig_label(uno_p),
        'uno_interp':        interpretation(uno_p),
        'uno_tau':           tau,
        # Coverage
        'subject_sample_count':     subject_sample_count,
        'n_subjects_ever_sampled':  n_ever_sampled,
        'n_subjects_never_sampled': n_never_sampled,
        'mean_sample_count':        mean_sample_count,
        'min_sample_count':         min_sample_count,
        'max_sample_count':         max_sample_count,
        'expected_mean_count':      expected_mean,
        'n_harrell_boots':          len(harrell_diff_boots),
        'n_uno_boots':              len(uno_diff_boots),
    }


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("\n[1/4] Loading data...")

radiomic_disc = pd.read_csv(RADIOMIC_DIR / 'data' / 'discovery_results.csv', index_col='SubjectID')
radiomic_rep  = pd.read_csv(RADIOMIC_DIR / 'data' / 'replicate_results.csv', index_col='SubjectID')
resnet_disc   = pd.read_csv(RESNET_DIR   / 'data' / 'discovery_results.csv', index_col='SubjectID')
resnet_rep    = pd.read_csv(RESNET_DIR   / 'data' / 'replicate_results.csv', index_col='SubjectID')

disc_data = prepare_data(radiomic_disc, resnet_disc, "Discovery")
rep_data  = prepare_data(radiomic_rep,  resnet_rep,  "Replicate")

print(f"  Discovery: {disc_data['n']} subjects ({disc_data['n_events']} events, "
      f"{disc_data['n_events']/disc_data['n']*100:.1f}%)")
print(f"  Replicate: {rep_data['n']} subjects ({rep_data['n_events']} events, "
      f"{rep_data['n_events']/rep_data['n']*100:.1f}%)")


# =============================================================================
# STEP 2: BOOTSTRAP C-INDEX COMPARISON
# =============================================================================

print("\n[2/4] Bootstrap C-index comparison...")

print("\n  Discovery cohort (bootstrap resamples from discovery):")
disc_boot = bootstrap_c_index_comparison(
    disc_data['event'], disc_data['time'],
    disc_data['rad_risk'], disc_data['res_risk'],
    disc_event=disc_data['event'], disc_time=disc_data['time'],
    n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED,
    cohort_label='Discovery'
)

print(f"\n    Harrell C — Radiomic: {disc_boot['harrell1_orig']:.4f} "
      f"[{disc_boot['harrell1_ci'][0]:.4f}, {disc_boot['harrell1_ci'][1]:.4f}]")
print(f"    Harrell C — ResNet:   {disc_boot['harrell2_orig']:.4f} "
      f"[{disc_boot['harrell2_ci'][0]:.4f}, {disc_boot['harrell2_ci'][1]:.4f}]")
print(f"    Δ Harrell C:  {disc_boot['harrell_diff_orig']:+.4f} "
      f"[{disc_boot['harrell_diff_ci'][0]:+.4f}, {disc_boot['harrell_diff_ci'][1]:+.4f}]  "
      f"p={disc_boot['harrell_p']:.4f} {disc_boot['harrell_sig']}")
print(f"    Uno C   — Radiomic: {disc_boot['uno1_orig']:.4f} "
      f"[{disc_boot['uno1_ci'][0]:.4f}, {disc_boot['uno1_ci'][1]:.4f}]")
print(f"    Uno C   — ResNet:   {disc_boot['uno2_orig']:.4f} "
      f"[{disc_boot['uno2_ci'][0]:.4f}, {disc_boot['uno2_ci'][1]:.4f}]")
print(f"    Δ Uno C:      {disc_boot['uno_diff_orig']:+.4f} "
      f"[{disc_boot['uno_diff_ci'][0]:+.4f}, {disc_boot['uno_diff_ci'][1]:+.4f}]  "
      f"p={disc_boot['uno_p']:.4f} {disc_boot['uno_sig']}")
print(f"    Uno tau: {disc_boot['uno_tau']:.2f} months "
      f"(90th pct of discovery event times)")

print("\n  Replicate cohort (bootstrap resamples from replicate; "
      "IPCW weights from discovery):")
rep_boot = bootstrap_c_index_comparison(
    rep_data['event'], rep_data['time'],
    rep_data['rad_risk'], rep_data['res_risk'],
    disc_event=disc_data['event'], disc_time=disc_data['time'],
    n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED,
    cohort_label='Replicate'
)

print(f"\n    Harrell C — Radiomic: {rep_boot['harrell1_orig']:.4f} "
      f"[{rep_boot['harrell1_ci'][0]:.4f}, {rep_boot['harrell1_ci'][1]:.4f}]")
print(f"    Harrell C — ResNet:   {rep_boot['harrell2_orig']:.4f} "
      f"[{rep_boot['harrell2_ci'][0]:.4f}, {rep_boot['harrell2_ci'][1]:.4f}]")
print(f"    Δ Harrell C:  {rep_boot['harrell_diff_orig']:+.4f} "
      f"[{rep_boot['harrell_diff_ci'][0]:+.4f}, {rep_boot['harrell_diff_ci'][1]:+.4f}]  "
      f"p={rep_boot['harrell_p']:.4f} {rep_boot['harrell_sig']}")
print(f"    Uno C   — Radiomic: {rep_boot['uno1_orig']:.4f} "
      f"[{rep_boot['uno1_ci'][0]:.4f}, {rep_boot['uno1_ci'][1]:.4f}]")
print(f"    Uno C   — ResNet:   {rep_boot['uno2_orig']:.4f} "
      f"[{rep_boot['uno2_ci'][0]:.4f}, {rep_boot['uno2_ci'][1]:.4f}]")
print(f"    Δ Uno C:      {rep_boot['uno_diff_orig']:+.4f} "
      f"[{rep_boot['uno_diff_ci'][0]:+.4f}, {rep_boot['uno_diff_ci'][1]:+.4f}]  "
      f"p={rep_boot['uno_p']:.4f} {rep_boot['uno_sig']}")
print(f"    Uno tau: {rep_boot['uno_tau']:.2f} months "
      f"(90th pct of discovery event times, applied to replicate)")


# =============================================================================
# STEP 3: VISUALIZATIONS
# =============================================================================

print("\n[3/4] Creating visualizations...")

# ── Figure 1: Harrell C distributions + difference distributions ──────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Bootstrap C-index Comparison: Clinical-Radiomic vs Clinical-ResNet\n"
             "Harrell's C-index", fontsize=14, fontweight='bold', y=0.995)

for idx, (cohort, boot, data) in enumerate([
        ('Discovery', disc_boot, disc_data),
        ('Replicate',  rep_boot,  rep_data)]):

    n_str = f"n={data['n']}, {data['n_events']} events"

    # Left: overlapping C-index histograms
    ax1 = axes[idx, 0]
    ax1.hist(boot['harrell1_boots'], bins=40, alpha=0.65, label='Clinical-Radiomic',
             color='#2E86AB', edgecolor='black', linewidth=0.5)
    ax1.hist(boot['harrell2_boots'], bins=40, alpha=0.65, label='Clinical-ResNet',
             color='#A23B72', edgecolor='black', linewidth=0.5)
    ax1.axvline(boot['harrell1_orig'], color='#2E86AB', linestyle='--',
                linewidth=2.5, alpha=0.9)
    ax1.axvline(boot['harrell2_orig'], color='#A23B72', linestyle='--',
                linewidth=2.5, alpha=0.9)
    ax1.set_xlabel("Harrell's C-index", fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'{cohort} ({n_str}): C-index Distribution',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper left')
    ax1.grid(alpha=0.25, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: difference distribution
    ax2 = axes[idx, 1]
    ax2.hist(boot['harrell_diff_boots'], bins=40, alpha=0.75,
             color='#06A77D', edgecolor='black', linewidth=0.5)
    ax2.axvline(boot['harrell_diff_orig'], color='#D62839', linestyle='--',
                linewidth=2.5, label=f"Observed Δ = {boot['harrell_diff_orig']:+.4f}")
    ax2.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.7,
                label='Null (Δ = 0)')
    ax2.axvline(boot['harrell_diff_ci'][0], color='#F77F00', linestyle=':', linewidth=2.5, alpha=0.85)
    ax2.axvline(boot['harrell_diff_ci'][1], color='#F77F00', linestyle=':', linewidth=2.5, alpha=0.85,
                label=f"95% CI [{boot['harrell_diff_ci'][0]:+.4f}, {boot['harrell_diff_ci'][1]:+.4f}]")

    ci_spans_zero = boot['harrell_diff_ci'][0] <= 0 <= boot['harrell_diff_ci'][1]
    ann = (f"p = {boot['harrell_p']:.4f}  {boot['harrell_sig']}\n"
           f"{'CI includes 0 — no significant difference' if ci_spans_zero else 'CI excludes 0'}")
    ax2.text(0.97, 0.97, ann, transform=ax2.transAxes,
             fontsize=10, fontweight='bold', va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       alpha=0.95, edgecolor='black', linewidth=1.5))

    ax2.set_xlabel('Δ C-index (Radiomic − ResNet)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'{cohort}: Bootstrap Difference (Harrell)',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.legend(frameon=True, fancybox=False, edgecolor='black',
               loc='upper left', fontsize=9)
    ax2.grid(alpha=0.25, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure1_Harrell_Bootstrap_Comparison.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure1_Harrell_Bootstrap_Comparison.tiff'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ── Figure 2: Uno's C distributions + difference distributions ───────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Bootstrap C-index Comparison: Clinical-Radiomic vs Clinical-ResNet\n"
             f"Uno's C-index (τ = {disc_boot['uno_tau']:.1f} months, "
             "90th pct of discovery event times)",
             fontsize=14, fontweight='bold', y=0.995)

for idx, (cohort, boot, data) in enumerate([
        ('Discovery', disc_boot, disc_data),
        ('Replicate',  rep_boot,  rep_data)]):

    n_str = f"n={data['n']}, {data['n_events']} events"

    ax1 = axes[idx, 0]
    if len(boot['uno1_boots']) > 0:
        ax1.hist(boot['uno1_boots'], bins=40, alpha=0.65, label='Clinical-Radiomic',
                 color='#2E86AB', edgecolor='black', linewidth=0.5)
        ax1.hist(boot['uno2_boots'], bins=40, alpha=0.65, label='Clinical-ResNet',
                 color='#A23B72', edgecolor='black', linewidth=0.5)
        ax1.axvline(boot['uno1_orig'], color='#2E86AB', linestyle='--', linewidth=2.5, alpha=0.9)
        ax1.axvline(boot['uno2_orig'], color='#A23B72', linestyle='--', linewidth=2.5, alpha=0.9)
    ax1.set_xlabel("Uno's C-index", fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'{cohort} ({n_str}): Uno C Distribution',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper left')
    ax1.grid(alpha=0.25, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = axes[idx, 1]
    if len(boot['uno_diff_boots']) > 0:
        ax2.hist(boot['uno_diff_boots'], bins=40, alpha=0.75,
                 color='#06A77D', edgecolor='black', linewidth=0.5)
        ax2.axvline(boot['uno_diff_orig'], color='#D62839', linestyle='--',
                    linewidth=2.5, label=f"Observed Δ = {boot['uno_diff_orig']:+.4f}")
        ax2.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.7,
                    label='Null (Δ = 0)')
        ax2.axvline(boot['uno_diff_ci'][0], color='#F77F00', linestyle=':', linewidth=2.5, alpha=0.85)
        ax2.axvline(boot['uno_diff_ci'][1], color='#F77F00', linestyle=':', linewidth=2.5, alpha=0.85,
                    label=f"95% CI [{boot['uno_diff_ci'][0]:+.4f}, {boot['uno_diff_ci'][1]:+.4f}]")

        ci_spans_zero = boot['uno_diff_ci'][0] <= 0 <= boot['uno_diff_ci'][1]
        ann = (f"p = {boot['uno_p']:.4f}  {boot['uno_sig']}\n"
               f"{'CI includes 0 — no significant difference' if ci_spans_zero else 'CI excludes 0'}")
        ax2.text(0.97, 0.97, ann, transform=ax2.transAxes,
                 fontsize=10, fontweight='bold', va='top', ha='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                           alpha=0.95, edgecolor='black', linewidth=1.5))

    ax2.set_xlabel("Δ Uno C (Radiomic − ResNet)", fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'{cohort}: Bootstrap Difference (Uno)',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.legend(frameon=True, fancybox=False, edgecolor='black',
               loc='upper left', fontsize=9)
    ax2.grid(alpha=0.25, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure2_Uno_Bootstrap_Comparison.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure2_Uno_Bootstrap_Comparison.tiff'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ── Figure 3: Bootstrap Coverage Validation ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Bootstrap Coverage Validation\n"
             "(cumulative: how many times each subject appeared across all iterations)",
             fontsize=13, fontweight='bold')

for idx, (cohort, boot, data) in enumerate([
        ('Discovery', disc_boot, disc_data),
        ('Replicate',  rep_boot,  rep_data)]):

    ax = axes[idx]
    ax.hist(boot['subject_sample_count'], bins=35, alpha=0.75,
            color='#06A77D', edgecolor='black', linewidth=0.8)
    ax.axvline(boot['mean_sample_count'], color='#D62839', linestyle='--',
               linewidth=2.5, label=f"Observed mean = {boot['mean_sample_count']:.1f}")
    ax.axvline(boot['expected_mean_count'], color='#2E86AB', linestyle='--',
               linewidth=2.5, label=f"Expected mean = {boot['expected_mean_count']:.1f}")

    stats_text = (f"Sampled ≥1×: {boot['n_subjects_ever_sampled']}/{data['n']} "
                  f"({boot['n_subjects_ever_sampled']/data['n']*100:.0f}%)\n"
                  f"Range: {boot['min_sample_count']}–{boot['max_sample_count']}")
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=11, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      alpha=0.95, edgecolor='black', linewidth=1.5))

    ax.set_xlabel('Cumulative appearances across all bootstrap iterations',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
    ax.set_title(f'{cohort} Cohort: Coverage Validation',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure3_Bootstrap_Coverage.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure3_Bootstrap_Coverage.tiff'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  Figures saved (PNG and TIFF, 300 DPI)")


# =============================================================================
# STEP 4: SAVE RESULTS
# =============================================================================

print("\n[4/4] Saving results...")

summary_rows = []
for cohort, data, boot in [
        ('Discovery', disc_data, disc_boot),
        ('Replicate',  rep_data,  rep_boot)]:
    summary_rows.append({
        'Cohort':                    cohort,
        'N':                         data['n'],
        'N_Events':                  data['n_events'],
        'Event_Rate_Pct':            f"{data['n_events']/data['n']*100:.1f}",
        # Harrell
        'Radiomic_Harrell':          f"{boot['harrell1_orig']:.4f}",
        'Radiomic_Harrell_CI':       f"[{boot['harrell1_ci'][0]:.4f}, {boot['harrell1_ci'][1]:.4f}]",
        'ResNet_Harrell':            f"{boot['harrell2_orig']:.4f}",
        'ResNet_Harrell_CI':         f"[{boot['harrell2_ci'][0]:.4f}, {boot['harrell2_ci'][1]:.4f}]",
        'Delta_Harrell':             f"{boot['harrell_diff_orig']:+.4f}",
        'Delta_Harrell_CI':          f"[{boot['harrell_diff_ci'][0]:+.4f}, {boot['harrell_diff_ci'][1]:+.4f}]",
        'Harrell_P_Value':           f"{boot['harrell_p']:.4f}",
        'Harrell_Sig':               boot['harrell_sig'],
        'Harrell_Interpretation':    boot['harrell_interp'],
        # Uno
        'Radiomic_Uno':              f"{boot['uno1_orig']:.4f}" if not np.isnan(boot['uno1_orig']) else 'N/A',
        'Radiomic_Uno_CI':           f"[{boot['uno1_ci'][0]:.4f}, {boot['uno1_ci'][1]:.4f}]" if not np.isnan(boot['uno1_ci'][0]) else 'N/A',
        'ResNet_Uno':                f"{boot['uno2_orig']:.4f}" if not np.isnan(boot['uno2_orig']) else 'N/A',
        'ResNet_Uno_CI':             f"[{boot['uno2_ci'][0]:.4f}, {boot['uno2_ci'][1]:.4f}]" if not np.isnan(boot['uno2_ci'][0]) else 'N/A',
        'Delta_Uno':                 f"{boot['uno_diff_orig']:+.4f}" if not np.isnan(boot['uno_diff_orig']) else 'N/A',
        'Delta_Uno_CI':              f"[{boot['uno_diff_ci'][0]:+.4f}, {boot['uno_diff_ci'][1]:+.4f}]" if not np.isnan(boot['uno_diff_ci'][0]) else 'N/A',
        'Uno_P_Value':               f"{boot['uno_p']:.4f}" if not np.isnan(boot['uno_p']) else 'N/A',
        'Uno_Sig':                   boot['uno_sig'],
        'Uno_Tau_Months':            f"{boot['uno_tau']:.2f}" if boot['uno_tau'] else 'N/A',
        'Uno_Interpretation':        boot['uno_interp'],
        # Coverage
        'N_Successful_Harrell_Boots': boot['n_harrell_boots'],
        'N_Successful_Uno_Boots':     boot['n_uno_boots'],
        'Subjects_Ever_Sampled':      boot['n_subjects_ever_sampled'],
        'Mean_Appearances':           f"{boot['mean_sample_count']:.1f}",
        'Expected_Mean_Appearances':  f"{boot['expected_mean_count']:.1f}",
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'bootstrap_results.csv'), index=False)
print(f"  bootstrap_results.csv saved")

# Text report
with open(os.path.join(OUTPUT_DIR, 'BOOTSTRAP_REPORT.txt'), 'w') as f:
    f.write("="*80 + "\n")
    f.write("BOOTSTRAP C-INDEX COMPARISON\n")
    f.write("Clinical-Radiomic vs Clinical-ResNet\n")
    f.write("="*80 + "\n\n")
    f.write(f"Method: Paired bootstrap resampling ({N_BOOTSTRAP} iterations, seed={RANDOM_SEED})\n")
    f.write(f"  - Same resampled indices applied to both models (paired comparison)\n")
    f.write(f"  - Discovery: bootstrap resamples from discovery patients\n")
    f.write(f"  - Replicate: bootstrap resamples from replicate patients "
            f"(IPCW weights from discovery)\n")
    f.write(f"  - Uno tau = 90th percentile of discovery event times "
            f"({disc_boot['uno_tau']:.2f} months)\n")
    f.write(f"  - P-values: two-sided bootstrap permutation test\n")
    f.write(f"  - CIs: 2.5th-97.5th percentile of bootstrap difference distribution\n\n")
    f.write("NOTE: Given small replicate cohort (n≈78), the 95% CI on Δ C-index\n")
    f.write("is the primary inferential quantity. P-values are reported for\n")
    f.write("completeness but should not be over-interpreted.\n\n")

    for cohort, data, boot in [
            ('Discovery', disc_data, disc_boot),
            ('Replicate',  rep_data,  rep_boot)]:
        f.write(f"\n{cohort.upper()} COHORT  (n={data['n']}, {data['n_events']} events, "
                f"{data['n_events']/data['n']*100:.1f}% event rate)\n")
        f.write("-"*80 + "\n")

        f.write(f"\nHarrell's C-index ({boot['n_harrell_boots']} successful bootstrap iterations):\n")
        f.write(f"  Clinical-Radiomic: {boot['harrell1_orig']:.4f} "
                f"[95% CI: {boot['harrell1_ci'][0]:.4f}, {boot['harrell1_ci'][1]:.4f}]  "
                f"SE={boot['harrell1_se']:.4f}\n")
        f.write(f"  Clinical-ResNet:   {boot['harrell2_orig']:.4f} "
                f"[95% CI: {boot['harrell2_ci'][0]:.4f}, {boot['harrell2_ci'][1]:.4f}]  "
                f"SE={boot['harrell2_se']:.4f}\n")
        f.write(f"  Δ (Radiomic − ResNet): {boot['harrell_diff_orig']:+.4f} "
                f"[95% CI: {boot['harrell_diff_ci'][0]:+.4f}, {boot['harrell_diff_ci'][1]:+.4f}]\n")
        f.write(f"  P-value: {boot['harrell_p']:.4f}  {boot['harrell_sig']}\n")
        f.write(f"  → {boot['harrell_interp']}\n")

        f.write(f"\nUno's C-index (tau={boot['uno_tau']:.2f} months, "
                f"{boot['n_uno_boots']} successful bootstrap iterations):\n")
        if not np.isnan(boot['uno1_orig']):
            f.write(f"  Clinical-Radiomic: {boot['uno1_orig']:.4f} "
                    f"[95% CI: {boot['uno1_ci'][0]:.4f}, {boot['uno1_ci'][1]:.4f}]  "
                    f"SE={boot['uno1_se']:.4f}\n")
            f.write(f"  Clinical-ResNet:   {boot['uno2_orig']:.4f} "
                    f"[95% CI: {boot['uno2_ci'][0]:.4f}, {boot['uno2_ci'][1]:.4f}]  "
                    f"SE={boot['uno2_se']:.4f}\n")
            f.write(f"  Δ (Radiomic − ResNet): {boot['uno_diff_orig']:+.4f} "
                    f"[95% CI: {boot['uno_diff_ci'][0]:+.4f}, {boot['uno_diff_ci'][1]:+.4f}]\n")
            f.write(f"  P-value: {boot['uno_p']:.4f}  {boot['uno_sig']}\n")
            f.write(f"  → {boot['uno_interp']}\n")
        else:
            f.write("  Uno C not available (IPCW computation failed)\n")

        f.write(f"\nBootstrap Coverage:\n")
        f.write(f"  Subjects sampled ≥1×: {boot['n_subjects_ever_sampled']}/{data['n']} "
                f"({boot['n_subjects_ever_sampled']/data['n']*100:.0f}%)\n")
        f.write(f"  Mean appearances per subject: {boot['mean_sample_count']:.1f}  "
                f"(expected ~{boot['expected_mean_count']:.1f})\n")
        f.write(f"  Range: [{boot['min_sample_count']}, {boot['max_sample_count']}]\n")

print(f"  BOOTSTRAP_REPORT.txt saved")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("BOOTSTRAP ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}\n")
print("Results Summary:\n")
for cohort, boot in [('Discovery', disc_boot), ('Replicate', rep_boot)]:
    print(f"  {cohort}:")
    print(f"    Δ Harrell C: {boot['harrell_diff_orig']:+.4f} "
          f"[{boot['harrell_diff_ci'][0]:+.4f}, {boot['harrell_diff_ci'][1]:+.4f}]  "
          f"p={boot['harrell_p']:.4f}  {boot['harrell_sig']}")
    print(f"    Δ Uno C:     {boot['uno_diff_orig']:+.4f} "
          f"[{boot['uno_diff_ci'][0]:+.4f}, {boot['uno_diff_ci'][1]:+.4f}]  "
          f"p={boot['uno_p']:.4f}  {boot['uno_sig']}")
    print(f"    → {boot['harrell_interp']}")
    print()

print("Output files:")
print("  Figure1_Harrell_Bootstrap_Comparison.png/tiff")
print("  Figure2_Uno_Bootstrap_Comparison.png/tiff")
print("  Figure3_Bootstrap_Coverage.png/tiff")
print("  bootstrap_results.csv")
print("  BOOTSTRAP_REPORT.txt")
print("="*80)