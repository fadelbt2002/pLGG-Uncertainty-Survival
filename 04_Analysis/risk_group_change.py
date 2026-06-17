#!/usr/bin/env python3
"""
Risk Group Transition Analysis: M1 -> M2 Pairwise (Clinico-ResNet only)

Analyzes patient-level risk group transitions:
- M1: Clinical-ResNet base model
- M2: ClinResNet + Molecular pairwise fusion

Author: Fadel Batal
Date: 2026-02-04
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Repo root: one level up from this script's folder (04_Analysis/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DLM1_DIR  = REPO_ROOT / '01_DLM1_Clinico_ResNet'
DLM2_DIR  = REPO_ROOT / '03_Late_Fusion_DLM2'

BASE_MODEL_PATHS = {
    'Clinical-ResNet': {
        'discovery': DLM1_DIR / 'outputs' / 'data' / 'discovery_results.csv',
        'replicate': DLM1_DIR / 'outputs' / 'data' / 'replicate_results.csv',
        'threshold': DLM1_DIR / 'outputs' / 'metrics' / 'risk_thresholds.txt',
    }
}

# Update FUSION_DIR to the timestamped folder produced by
# 03_Late_Fusion_DLM2/Late_Fusion_Modeling_ResNet.py
FUSION_DIR = DLM2_DIR / 'outputs' / 'pairwise_resnet_molecular_YYYYMMDD_HHMMSS'

PAIRWISE_MODEL_PATHS = {
    'ClinResNet+Mol': {
        'discovery': FUSION_DIR / 'risk_scores_discovery.csv',
        'replicate': FUSION_DIR / 'risk_scores_replicate.csv',
    }
}

OUTPUT_DIR = str(REPO_ROOT / '04_Analysis' / 'outputs' / 'risk_group_transitions')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQUENCE = {
    'M1': 'Clinical-ResNet',
    'M2': 'ClinResNet+Mol'
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_threshold_file(filepath):
    """Extract risk threshold value from model output file."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if '(50th percentile/median):' in line:
                    return float(line.split(':')[1].strip())
    except Exception as e:
        print(f"  Warning: Could not parse {filepath}: {e}")
        return None


def apply_risk_stratification(df, threshold):
    """Apply binary risk stratification based on threshold."""
    df['Risk Group'] = df['Risk Score'].apply(
        lambda x: 'High' if x > threshold else 'Low'
    )
    return df


def analyze_pairwise_m1_m2_transitions(cohort_name, base_models, pairwise_models):
    """
    Analyze risk group transitions between M1 and M2.
    Uses maximum available patients with both M1 and M2 data.
    """

    print(f"\n  Analyzing ResNet M1->M2 pairwise - {cohort_name} cohort")

    m1_name = SEQUENCE['M1']
    m2_name = SEQUENCE['M2']

    m1_data = base_models[m1_name][cohort_name].copy()
    m2_data = pairwise_models[m2_name][cohort_name].copy()

    m1_threshold = base_models[m1_name]['threshold']
    m2_threshold = pairwise_models[m2_name]['threshold']

    print(f"    M1: {len(m1_data)} patients (threshold={m1_threshold:.6f})")
    print(f"    M2: {len(m2_data)} patients (threshold={m2_threshold:.6f})")

    m1_data = m1_data.set_index('SubjectID')
    m2_data = m2_data.set_index('SubjectID')

    common_subjects = sorted(list(set(m1_data.index) & set(m2_data.index)))
    n_common = len(common_subjects)
    print(f"    Common patients (M1∩M2): {n_common}")

    if n_common == 0:
        print("    Warning: No overlapping patients found")
        return pd.DataFrame(), {}

    transition_records = []

    for subject_id in common_subjects:
        record = {
            'SubjectID':     subject_id,
            'Cohort':        cohort_name.capitalize(),
            'Analysis_Type': 'Pairwise_M1_M2'
        }

        record['PFS_Months']   = m1_data.loc[subject_id, 'Progression Free Survival']
        record['Event']        = m1_data.loc[subject_id, 'Event']
        record['Event_Status'] = 'Progressed' if record['Event'] else 'Censored'

        m1_risk  = m1_data.loc[subject_id, 'Risk Score']
        m1_group = m1_data.loc[subject_id, 'Risk Group']
        record['M1_Model']                   = m1_name
        record['M1_Risk_Score']              = m1_risk
        record['M1_Threshold']               = m1_threshold
        record['M1_Distance_from_Threshold'] = m1_risk - m1_threshold
        record['M1_Risk_Group']              = m1_group
        record['M1_Cohort_Size']             = len(m1_data)

        m2_risk  = m2_data.loc[subject_id, 'Risk Score']
        m2_group = m2_data.loc[subject_id, 'Risk Group']
        record['M2_Model']                   = m2_name
        record['M2_Risk_Score']              = m2_risk
        record['M2_Threshold']               = m2_threshold
        record['M2_Distance_from_Threshold'] = m2_risk - m2_threshold
        record['M2_Risk_Group']              = m2_group
        record['M2_Cohort_Size']             = len(m2_data)

        if m1_group == m2_group:
            record['Transition_M1_to_M2']        = f'Stable_{m1_group}'
            record['Transition_M1_to_M2_Status'] = 'Stable'
        else:
            record['Transition_M1_to_M2']        = f'{m1_group}_to_{m2_group}'
            record['Transition_M1_to_M2_Status'] = 'Changed'

        record['Overall_Trajectory'] = f'{m1_group}_{m2_group}'
        record['Any_Risk_Change']    = 'Yes' if m1_group != m2_group else 'No'

        transition_records.append(record)

    transitions_df = pd.DataFrame(transition_records)

    n_changed   = (transitions_df['Any_Risk_Change'] == 'Yes').sum()
    pct_changed = n_changed / n_common * 100

    print(f"    Transitions observed:")
    print(f"      M1->M2: {n_changed}/{n_common} ({pct_changed:.1f}%)")

    threshold_info = {
        'M1': {'name': m1_name, 'threshold': m1_threshold, 'n': len(m1_data)},
        'M2': {'name': m2_name, 'threshold': m2_threshold, 'n': len(m2_data)}
    }

    return transitions_df, threshold_info


def create_risk_transition_figure(merged_data, output_dir):
    """
    Create risk score transition visualization with confidence intervals
    for Clinico-ResNet M1 -> M2 pairwise analysis.
    """

    if len(merged_data) == 0:
        print("  No data available for figure")
        return None

    fig, ax = plt.subplots(figsize=(20, 11))

    n_patients  = len(merged_data)
    x_positions = np.arange(n_patients)

    m1_threshold = merged_data.iloc[0]['M1_Threshold']
    m2_threshold = merged_data.iloc[0]['M2_Threshold']

    color_m1          = '#1f77b4'
    color_m2          = '#ff7f0e'
    color_low_to_high = '#d62728'
    color_high_to_low = '#2ca02c'

    offset = 0.15

    for idx, (_, row) in enumerate(merged_data.iterrows()):
        x_pos = idx

        m1_risk     = row['M1_Risk_Score']
        m2_risk     = row['M2_Risk_Score']
        m1_ci_width = row['M1_Risk_CI_Width']
        m2_ci_width = row['M2_Risk_CI_Width']

        m1_error = m1_ci_width / 2
        m2_error = m2_ci_width / 2

        transition = row['Transition_M1_to_M2']
        bg_color   = color_low_to_high if 'Low_to_High' in transition else color_high_to_low

        ax.axvspan(x_pos - 0.4, x_pos + 0.4, facecolor=bg_color, alpha=0.1, zorder=1)

        ax.errorbar(x_pos - offset, m1_risk, yerr=m1_error,
                    fmt='o', markersize=12, color=color_m1,
                    ecolor=color_m1, elinewidth=3, capsize=8, capthick=3,
                    label='M1 (Clinical-ResNet)' if idx == 0 else '',
                    zorder=3, alpha=0.8)

        ax.errorbar(x_pos + offset, m2_risk, yerr=m2_error,
                    fmt='s', markersize=12, color=color_m2,
                    ecolor=color_m2, elinewidth=3, capsize=8, capthick=3,
                    label='M2 (ClinResNet + Molecular)' if idx == 0 else '',
                    zorder=3, alpha=0.8)

        ax.plot([x_pos - offset, x_pos + offset], [m1_risk, m2_risk],
                color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=2)

        ax.text(x_pos - offset, m1_risk - m1_error - 0.15,
                f'CI: {m1_ci_width:.2f}',
                ha='center', va='top', fontsize=7.5, color=color_m1,
                fontweight='bold', alpha=0.9)
        ax.text(x_pos + offset, m2_risk - m2_error - 0.15,
                f'CI: {m2_ci_width:.2f}',
                ha='center', va='top', fontsize=7.5, color=color_m2,
                fontweight='bold', alpha=0.9)

    ax.axhline(m1_threshold, color=color_m1, linestyle='--', linewidth=3,
               alpha=0.6, zorder=2, label=f'M1 Threshold ({m1_threshold:.3f})')
    ax.axhline(m2_threshold, color=color_m2, linestyle='--', linewidth=3,
               alpha=0.6, zorder=2, label=f'M2 Threshold ({m2_threshold:.3f})')

    all_values = []
    for _, row in merged_data.iterrows():
        all_values.extend([
            row['M1_Risk_Score'] - row['M1_Risk_CI_Width'] / 2,
            row['M1_Risk_Score'] + row['M1_Risk_CI_Width'] / 2,
            row['M2_Risk_Score'] - row['M2_Risk_CI_Width'] / 2,
            row['M2_Risk_Score'] + row['M2_Risk_CI_Width'] / 2
        ])

    y_min = min(all_values) - 0.5
    y_max = max(all_values) + 0.2
    ax.set_ylim(y_min, y_max)

    ax.fill_between([-0.5, n_patients - 0.5], m1_threshold, y_max,
                    color='red', alpha=0.05, zorder=0, label='High Risk Zone')
    ax.fill_between([-0.5, n_patients - 0.5], y_min, m1_threshold,
                    color='green', alpha=0.05, zorder=0, label='Low Risk Zone')

    ax.set_xlim(-0.5, n_patients - 0.5)
    ax.set_xlabel('Patient ID (Cohort) [Transition Direction]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Risk Score (with 95% CI)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Risk Group Transitions with Uncertainty Quantification: Clinico-ResNet\n'
        'M1 (Clinical-ResNet) → M2 (ClinResNet + Molecular)\n'
        '(D)=Discovery, (R)=Replicate, ↑=Low→High, ↓=High→Low',
        fontsize=16, fontweight='bold', pad=20
    )

    patient_labels = []
    for _, row in merged_data.iterrows():
        cohort_abbr      = 'D' if str(row['Cohort']).lower() == 'discovery' else 'R'
        transition_arrow = '↑' if 'Low_to_High' in row['Transition_M1_to_M2'] else '↓'
        patient_labels.append(f"{row['SubjectID']}\n({cohort_abbr}) {transition_arrow}")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(patient_labels, fontsize=10, fontweight='bold')

    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_m1,
               markersize=10, markeredgecolor='black', label='M1 (Clinical-ResNet)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_m2,
               markersize=10, markeredgecolor='black', label='M2 (ClinResNet + Molecular)'),
        Line2D([0], [0], color=color_m1, linestyle='--', linewidth=2,
               label=f'M1 Threshold ({m1_threshold:.3f})'),
        Line2D([0], [0], color=color_m2, linestyle='--', linewidth=2,
               label=f'M2 Threshold ({m2_threshold:.3f})'),
        Line2D([0], [0], color='red',   linewidth=5, alpha=0.3, label='High Risk Zone'),
        Line2D([0], [0], color='green', linewidth=5, alpha=0.3, label='Low Risk Zone'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', framealpha=0.95, fontsize=10,
              ncol=3, title='Error bars = 95% CI width', title_fontsize=10,
              bbox_to_anchor=(0.5, -0.12))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    figure_path = os.path.join(output_dir, 'FIGURE_Risk_Transitions_Pairwise_ResNet.png')
    fig.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    print(f"  Figure saved: {figure_path}")
    plt.close()

    return figure_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 80)
    print("Patient-Level Risk Group Transition Analysis")
    print("Clinico-ResNet: M1 -> M2 Pairwise")
    print("=" * 80)

    # ── Step 1: Load M1 (Clinical-ResNet base model) ──────────────────────────
    print("\nStep 1: Loading base model (M1 - Clinical-ResNet)...")

    base_models = {}
    model_name  = 'Clinical-ResNet'
    paths       = BASE_MODEL_PATHS[model_name]

    discovery_df = pd.read_csv(paths['discovery'])
    replicate_df = pd.read_csv(paths['replicate'])

    if 'SubjectID' not in discovery_df.columns:
        discovery_df = discovery_df.rename(columns={discovery_df.columns[0]: 'SubjectID'})
    if 'SubjectID' not in replicate_df.columns:
        replicate_df = replicate_df.rename(columns={replicate_df.columns[0]: 'SubjectID'})

    threshold = parse_threshold_file(paths['threshold'])
    if threshold is None:
        threshold = discovery_df['Risk Score'].median()
        print(f"  Using calculated discovery median: {threshold:.6f}")
    else:
        print(f"  Threshold: {threshold:.6f}")

    discovery_df = apply_risk_stratification(discovery_df, threshold)
    replicate_df = apply_risk_stratification(replicate_df, threshold)

    base_models[model_name] = {
        'discovery': discovery_df,
        'replicate': replicate_df,
        'threshold': threshold,
        'n_discovery': len(discovery_df),
        'n_replicate': len(replicate_df)
    }

    print(f"  Discovery: {len(discovery_df)} patients")
    print(f"  Replicate: {len(replicate_df)} patients")

    # ── Step 2: Load M2 (ClinResNet + Molecular pairwise fusion) ──────────────
    print("\nStep 2: Loading pairwise fusion model (M2 - ClinResNet + Molecular)...")

    pairwise_models = {}
    m2_name         = 'ClinResNet+Mol'
    m2_paths        = PAIRWISE_MODEL_PATHS[m2_name]

    m2_disc = pd.read_csv(m2_paths['discovery'])
    m2_rep  = pd.read_csv(m2_paths['replicate'])

    m2_threshold = m2_disc['Risk Score'].median()
    print(f"  Threshold (discovery median): {m2_threshold:.6f}")
    print(f"  Discovery: {len(m2_disc)} patients")
    print(f"  Replicate: {len(m2_rep)} patients")

    m2_disc = apply_risk_stratification(m2_disc, m2_threshold)
    m2_rep  = apply_risk_stratification(m2_rep,  m2_threshold)

    pairwise_models[m2_name] = {
        'discovery': m2_disc,
        'replicate': m2_rep,
        'threshold': m2_threshold,
        'n_discovery': len(m2_disc),
        'n_replicate': len(m2_rep)
    }

    # ── Step 3: Run transition analysis ───────────────────────────────────────
    print("\nStep 3: Running transition analyses...")

    results = {}
    for cohort_name in ['discovery', 'replicate']:
        key = f"{cohort_name.capitalize()}_ResNet_Pairwise_M1M2"
        transitions_df, threshold_info = analyze_pairwise_m1_m2_transitions(
            cohort_name, base_models, pairwise_models
        )
        results[key] = {
            'transitions':   transitions_df,
            'thresholds':    threshold_info,
            'analysis_type': 'Pairwise_M1_M2'
        }

    # ── Step 4: Export Excel ───────────────────────────────────────────────────
    print("\nStep 4: Exporting results to Excel...")

    column_order = [
        'SubjectID', 'Cohort', 'Analysis_Type',
        'PFS_Months', 'Event', 'Event_Status',
        'M1_Model', 'M1_Cohort_Size', 'M1_Risk_Score', 'M1_Threshold',
        'M1_Distance_from_Threshold', 'M1_Risk_Group',
        'M2_Model', 'M2_Cohort_Size', 'M2_Risk_Score', 'M2_Threshold',
        'M2_Distance_from_Threshold', 'M2_Risk_Group',
        'Transition_M1_to_M2', 'Transition_M1_to_M2_Status',
        'Overall_Trajectory', 'Any_Risk_Change'
    ]

    for key, result in results.items():
        transitions_df = result['transitions']
        threshold_info = result['thresholds']

        if len(transitions_df) == 0:
            continue

        excel_path = os.path.join(OUTPUT_DIR, f'patient_transitions_{key}.xlsx')

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            existing_cols = [c for c in column_order if c in transitions_df.columns]
            remaining     = [c for c in transitions_df.columns if c not in column_order]
            transitions_df[existing_cols + remaining].to_excel(
                writer, sheet_name='Patient_Data', index=False)

            pd.DataFrame([
                {'Stage': 'M1', 'Model_Name': threshold_info['M1']['name'],
                 'Original_Cohort_N': threshold_info['M1']['n'],
                 'Risk_Threshold': threshold_info['M1']['threshold'],
                 'Classification_Rule': f"Risk Score > {threshold_info['M1']['threshold']:.6f} = High Risk",
                 'Note': 'Threshold from risk_thresholds.txt (50th percentile)'},
                {'Stage': 'M2', 'Model_Name': threshold_info['M2']['name'],
                 'Original_Cohort_N': threshold_info['M2']['n'],
                 'Risk_Threshold': threshold_info['M2']['threshold'],
                 'Classification_Rule': f"Risk Score > {threshold_info['M2']['threshold']:.6f} = High Risk",
                 'Note': 'Threshold = discovery median'}
            ]).to_excel(writer, sheet_name='Model_Thresholds', index=False)

            changed = transitions_df[transitions_df['Transition_M1_to_M2_Status'] == 'Changed']
            if len(changed) > 0:
                changed.to_excel(writer, sheet_name='M1_to_M2_Changes', index=False)

            n_total   = len(transitions_df)
            n_changed = (transitions_df['Any_Risk_Change'] == 'Yes').sum()
            n_events  = (transitions_df['Event'] == True).sum()

            pd.DataFrame({
                'Metric': [
                    'Analysis Type', 'Common Patients (M1∩M2)',
                    'M1 Original Cohort N', 'M2 Original Cohort N', '',
                    'M1 Threshold', 'M2 Threshold', '',
                    'M1->M2 Changes', 'Stable Patients', 'Pct Changed', '',
                    'Progression Events', 'Event Rate',
                    'Most Common Trajectory'
                ],
                'Value': [
                    'Pairwise M1→M2', n_total,
                    threshold_info['M1']['n'], threshold_info['M2']['n'], '',
                    f"{threshold_info['M1']['threshold']:.6f}",
                    f"{threshold_info['M2']['threshold']:.6f}", '',
                    n_changed, n_total - n_changed,
                    f"{n_changed / n_total * 100:.1f}%", '',
                    n_events, f"{n_events / n_total * 100:.1f}%",
                    transitions_df['Overall_Trajectory'].mode().values[0] if n_total > 0 else 'N/A'
                ]
            }).to_excel(writer, sheet_name='Summary', index=False)

        print(f"  Saved: patient_transitions_{key}.xlsx")

    # Master combined file
    print("\n  Creating master combined file...")

    all_dfs = [r['transitions'] for r in results.values() if len(r['transitions']) > 0]
    if all_dfs:
        master_path = os.path.join(OUTPUT_DIR, 'MASTER_All_Patient_Transitions.xlsx')
        with pd.ExcelWriter(master_path, engine='openpyxl') as writer:
            combined  = pd.concat(all_dfs, ignore_index=True)
            existing  = [c for c in column_order if c in combined.columns]
            remaining = [c for c in combined.columns if c not in column_order]
            combined[existing + remaining].to_excel(writer, sheet_name='All_Combined', index=False)

            for key, result in results.items():
                if len(result['transitions']) > 0:
                    df = result['transitions']
                    ex = [c for c in column_order if c in df.columns]
                    re = [c for c in df.columns if c not in column_order]
                    df[ex + re].to_excel(writer, sheet_name=key[:31], index=False)

        print(f"  Saved: MASTER_All_Patient_Transitions.xlsx")

    # ── Step 5: Visualizations ─────────────────────────────────────────────────
    print("\nStep 5: Generating transition figure...")

    uncertainty_file = str(REPO_ROOT / '04_Analysis' / 'outputs' / 'uncertainty_DLM1_vs_DLM2' / 'COMPARISON_patient_level.csv')
    master_file      = os.path.join(OUTPUT_DIR, 'MASTER_All_Patient_Transitions.xlsx')

    if not os.path.exists(master_file):
        print("  Master file not found — skipping figure")
    elif not os.path.exists(uncertainty_file):
        print(f"  Uncertainty file not found: {uncertainty_file}")
    else:
        all_transitions = pd.read_excel(master_file, sheet_name='All_Combined')
        uncertainty_df  = pd.read_csv(uncertainty_file)

        changed_patients = all_transitions[
            all_transitions['Transition_M1_to_M2_Status'] == 'Changed'
        ].copy()

        merged_data = changed_patients.merge(
            uncertainty_df[['SubjectID', 'M1_Risk_CI_Width', 'M2_Risk_CI_Width']],
            on='SubjectID', how='left'
        )

        merged_data['Cohort_Lower'] = merged_data['Cohort'].str.lower()
        merged_data = merged_data.sort_values(['Cohort_Lower', 'SubjectID'])

        print(f"  Patients with transitions: {len(merged_data)}")
        print(f"    Discovery: {(merged_data['Cohort_Lower'] == 'discovery').sum()}")
        print(f"    Replicate: {(merged_data['Cohort_Lower'] == 'replicate').sum()}")

        if len(merged_data) > 0:
            create_risk_transition_figure(merged_data, OUTPUT_DIR)

            summary = merged_data[[
                'SubjectID', 'Cohort', 'Transition_M1_to_M2',
                'M1_Risk_Score', 'M1_Risk_CI_Width', 'M1_Threshold', 'M1_Distance_from_Threshold',
                'M2_Risk_Score', 'M2_Risk_CI_Width', 'M2_Threshold', 'M2_Distance_from_Threshold',
                'PFS_Months', 'Event_Status'
            ]].copy()

            summary['M1_CI_Crosses_Threshold'] = summary.apply(
                lambda row: (
                    (row['M1_Risk_Score'] - row['M1_Risk_CI_Width'] / 2 < row['M1_Threshold']) and
                    (row['M1_Risk_Score'] + row['M1_Risk_CI_Width'] / 2 > row['M1_Threshold'])
                ), axis=1
            )
            summary['M2_CI_Crosses_Threshold'] = summary.apply(
                lambda row: (
                    (row['M2_Risk_Score'] - row['M2_Risk_CI_Width'] / 2 < row['M2_Threshold']) and
                    (row['M2_Risk_Score'] + row['M2_Risk_CI_Width'] / 2 > row['M2_Threshold'])
                ), axis=1
            )
            summary['Uncertainty_Reduction_Pct'] = (
                (summary['M1_Risk_CI_Width'] - summary['M2_Risk_CI_Width']) /
                summary['M1_Risk_CI_Width'] * 100
            ).round(2)

            table_path = os.path.join(OUTPUT_DIR, 'TABLE_Uncertainty_Summary_Pairwise_ResNet.xlsx')
            summary.to_excel(table_path, index=False)
            print(f"  Summary table saved: {table_path}")

            n_m1_crosses  = summary['M1_CI_Crosses_Threshold'].sum()
            n_m2_crosses  = summary['M2_CI_Crosses_Threshold'].sum()
            avg_reduction = summary['Uncertainty_Reduction_Pct'].mean()

            print(f"\n  Uncertainty Analysis:")
            print(f"    M1 CIs crossing threshold: {n_m1_crosses}/{len(summary)} "
                  f"({n_m1_crosses / len(summary) * 100:.1f}%)")
            print(f"    M2 CIs crossing threshold: {n_m2_crosses}/{len(summary)} "
                  f"({n_m2_crosses / len(summary) * 100:.1f}%)")
            print(f"    Average uncertainty reduction: {avg_reduction:.1f}%")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nMethodology:")
    print("  M1 threshold : from risk_thresholds.txt (50th percentile/median)")
    print("  M2 threshold : discovery cohort median")
    print("  Patient selection: intersection of M1 and M2 (maximum overlap)")
    print("  Risk classification: Binary stratification (High vs. Low risk)")
    print("=" * 80)


if __name__ == "__main__":
    main()