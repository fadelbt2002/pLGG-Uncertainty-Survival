"""
evaluate_segmentation.py
------------------------
Evaluates the final segmentation model (output of train_fold_with_cv.py)
on a held-out test set.

Outputs:
  - test_evaluation_results.json   — full metrics + per-sample breakdown
  - test_per_sample_results.csv    — one row per test volume
  - test_evaluation_report.txt     — human-readable summary + paper-ready line
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats
from tqdm import tqdm

from dataset import CombinedSegmentationDataset
import model as model_utils


# ===========================================================================
# METRICS
# ===========================================================================

def calculate_metrics(logits, target, threshold=0.5):
    """Per-volume binary segmentation metrics from raw logits."""
    pred = (torch.sigmoid(logits) > threshold).float()

    p = pred.view(-1)
    t = target.view(-1)

    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    tn = ((1 - p) * (1 - t)).sum()
    fn = ((1 - p) * t).sum()

    eps = 1e-8
    dice        = (2 * tp) / (2 * tp + fp + fn + eps)
    iou         = tp / (tp + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    precision   = tp / (tp + fp + eps)
    f1          = 2 * (precision * sensitivity) / (precision + sensitivity + eps)

    return {
        'dice':        dice.item(),
        'iou':         iou.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
        'precision':   precision.item(),
        'f1':          f1.item(),
    }


def calculate_95_ci(values):
    n     = len(values)
    mean  = np.mean(values)
    std   = np.std(values, ddof=1)
    sem   = std / np.sqrt(n)
    t_val = stats.t.ppf(0.975, n - 1)
    return mean, mean - t_val * sem, mean + t_val * sem, std


# ===========================================================================
# MODEL LOADING
# ===========================================================================

def build_model_from_checkpoint(ckpt_path, input_size, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    hp         = checkpoint['hyperparams']
    depth      = hp.get('model_depth', 34)

    class _Opt:
        model           = 'resnet'
        resnet_shortcut = 'B'
        no_cuda         = False
        gpu_id          = [0]
        phase           = 'test'
        n_seg_classes   = 1
        new_layer_names = ['conv_seg']
        pretrain_path   = None

        def __init__(self, depth, sz):
            self.model_depth = depth
            self.input_D, self.input_H, self.input_W = sz

    m, _ = model_utils.generate_model(_Opt(depth, input_size))
    m.load_state_dict(checkpoint['model_state_dict'])
    m.to(device).eval()
    return m, checkpoint


# ===========================================================================
# EVALUATION LOOP
# ===========================================================================

def evaluate(model, loader, device):
    all_metrics = []

    with torch.no_grad():
        for images, masks, meta in tqdm(loader, desc="  evaluating"):
            try:
                images = images.to(device)
                masks  = masks.to(device)

                out = model(images)
                if out.shape != masks.shape:
                    out = nn.functional.interpolate(
                        out, size=masks.shape[2:],
                        mode='trilinear', align_corners=False)

                for i in range(images.shape[0]):
                    m = calculate_metrics(out[i:i+1], masks[i:i+1])
                    m['subject']  = meta['subject'][i]
                    m['session']  = str(meta['session'][i])
                    m['modality'] = meta['modality'][i]
                    all_metrics.append(m)

            except RuntimeError as e:
                print(f"  [WARN] batch error: {e}")
                torch.cuda.empty_cache()

    return all_metrics


# ===========================================================================
# MAIN
# ===========================================================================

def run_evaluation(args):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Test dataset ----
    dataset = CombinedSegmentationDataset(args.test_data_file)
    if args.modality:
        dataset = dataset.filter_by_modality(args.modality)

    print(f"Test samples: {len(dataset)}")
    print(f"Unique subjects: {dataset.df['Subject'].nunique()}")

    sample_img, _, _ = dataset[0]
    input_size = sample_img.shape[1:]
    print(f"Input size: {input_size}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # ---- Locate model checkpoints ----
    # Supports two layouts produced by train_fold_with_cv.py:
    #   1) single final_model.pth directly in experiment_dir
    #   2) fold_X/final_model.pth subdirectories (future / nested CV)
    experiment_dir = Path(args.experiment_dir)
    model_paths = []

    single = experiment_dir / "final_model.pth"
    if single.exists():
        model_paths = [("final", single)]
    else:
        fold_dirs = sorted(experiment_dir.glob("fold_*/final_model.pth"))
        model_paths = [(p.parent.name, p) for p in fold_dirs]

    if not model_paths:
        raise FileNotFoundError(
            f"No final_model.pth found in {experiment_dir} or its fold_* subdirs.")

    print(f"\nFound {len(model_paths)} model(s): {[n for n,_ in model_paths]}")

    # ---- Evaluate each model ----
    all_model_results = []

    for model_name, ckpt_path in model_paths:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}  |  {ckpt_path}")
        print(f"{'='*70}")

        model, ckpt = build_model_from_checkpoint(ckpt_path, input_size, device)

        hp = ckpt['hyperparams']
        print(f"  depth={hp.get('model_depth',34)}, "
              f"lr={hp.get('learning_rate','?')}, "
              f"bs={hp.get('batch_size','?')}, "
              f"wd={hp.get('weight_decay','?')}")
        print(f"  train_samples={ckpt.get('training_samples','?')}, "
              f"best_train_loss={ckpt.get('best_train_loss', float('nan')):.4f}")

        metrics = evaluate(model, loader, device)

        dice_vals = [m['dice']        for m in metrics]
        iou_vals  = [m['iou']         for m in metrics]
        sens_vals = [m['sensitivity'] for m in metrics]
        spec_vals = [m['specificity'] for m in metrics]
        prec_vals = [m['precision']   for m in metrics]
        f1_vals   = [m['f1']          for m in metrics]

        def _stats(vals):
            mean, lo, hi, std = calculate_95_ci(vals)
            return {'mean': mean, 'std': std, 'ci_lower': lo, 'ci_upper': hi,
                    'median': float(np.median(vals)),
                    'min': float(np.min(vals)), 'max': float(np.max(vals))}

        result = {
            'model':      model_name,
            'ckpt_path':  str(ckpt_path),
            'n_samples':  len(metrics),
            'dice':        _stats(dice_vals),
            'iou':         _stats(iou_vals),
            'sensitivity': _stats(sens_vals),
            'specificity': _stats(spec_vals),
            'precision':   _stats(prec_vals),
            'f1':          _stats(f1_vals),
            'per_sample':  metrics,
        }
        all_model_results.append(result)

        print(f"\n  Dice:        {result['dice']['mean']:.4f} ± {result['dice']['std']:.4f} "
              f"[{result['dice']['ci_lower']:.4f}, {result['dice']['ci_upper']:.4f}]")
        print(f"  IoU:         {result['iou']['mean']:.4f} ± {result['iou']['std']:.4f}")
        print(f"  Sensitivity: {result['sensitivity']['mean']:.4f} ± {result['sensitivity']['std']:.4f}")
        print(f"  Specificity: {result['specificity']['mean']:.4f} ± {result['specificity']['std']:.4f}")
        print(f"  Precision:   {result['precision']['mean']:.4f} ± {result['precision']['std']:.4f}")
        print(f"  F1:          {result['f1']['mean']:.4f} ± {result['f1']['std']:.4f}")

        del model
        torch.cuda.empty_cache()

    # ---- Aggregate across models (if multiple) ----
    agg = {}
    for metric in ['dice', 'iou', 'sensitivity', 'specificity', 'precision', 'f1']:
        means = [r[metric]['mean'] for r in all_model_results]
        m, lo, hi, std = calculate_95_ci(means)
        agg[metric] = {'mean': m, 'std': std, 'ci_lower': lo, 'ci_upper': hi,
                       'per_model_means': means}

    # ---- Save JSON ----
    results_dict = {
        'timestamp':        timestamp,
        'experiment_dir':   str(experiment_dir),
        'test_data_file':   args.test_data_file,
        'modality':         args.modality,
        'n_models':         len(all_model_results),
        'n_test_samples':   len(dataset),
        'aggregated':       agg,
        'per_model_results': [
            {k: v for k, v in r.items() if k != 'per_sample'}
            for r in all_model_results
        ],
    }

    json_path = output_dir / f"test_evaluation_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✓ JSON results  → {json_path}")

    # ---- Save per-sample CSV ----
    rows = []
    for r in all_model_results:
        for s in r['per_sample']:
            s['model'] = r['model']
            rows.append(s)
    df = pd.DataFrame(rows)
    csv_path = output_dir / f"test_per_sample_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Per-sample CSV → {csv_path}")

    # ---- Save text report ----
    d  = agg['dice']
    io = agg['iou']
    se = agg['sensitivity']
    sp = agg['specificity']
    pr = agg['precision']
    f1 = agg['f1']

    report_path = output_dir / f"test_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SEGMENTATION TEST SET EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp:      {timestamp}\n")
        f.write(f"Experiment:     {experiment_dir}\n")
        f.write(f"Test CSV:       {args.test_data_file}\n")
        f.write(f"Modality:       {args.modality}\n")
        f.write(f"Test samples:   {len(dataset)}\n")
        f.write(f"Models:         {len(all_model_results)}\n\n")

        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Dice:        {d['mean']:.4f} ± {d['std']:.4f}  "
                f"(95% CI [{d['ci_lower']:.4f}, {d['ci_upper']:.4f}])\n")
        f.write(f"IoU:         {io['mean']:.4f} ± {io['std']:.4f}  "
                f"(95% CI [{io['ci_lower']:.4f}, {io['ci_upper']:.4f}])\n")
        f.write(f"Sensitivity: {se['mean']:.4f} ± {se['std']:.4f}  "
                f"(95% CI [{se['ci_lower']:.4f}, {se['ci_upper']:.4f}])\n")
        f.write(f"Specificity: {sp['mean']:.4f} ± {sp['std']:.4f}  "
                f"(95% CI [{sp['ci_lower']:.4f}, {sp['ci_upper']:.4f}])\n")
        f.write(f"Precision:   {pr['mean']:.4f} ± {pr['std']:.4f}  "
                f"(95% CI [{pr['ci_lower']:.4f}, {pr['ci_upper']:.4f}])\n")
        f.write(f"F1:          {f1['mean']:.4f} ± {f1['std']:.4f}  "
                f"(95% CI [{f1['ci_lower']:.4f}, {f1['ci_upper']:.4f}])\n\n")

        if len(all_model_results) > 1:
            f.write("PER-MODEL BREAKDOWN\n")
            f.write("-" * 70 + "\n")
            for r in all_model_results:
                f.write(f"\n  {r['model']}:\n")
                for metric in ['dice', 'iou', 'sensitivity', 'specificity', 'precision', 'f1']:
                    v = r[metric]
                    f.write(f"    {metric:<12} {v['mean']:.4f} ± {v['std']:.4f}"
                            f"  [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("PAPER-READY LINE\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"The 3D ResNet-34 segmentation model achieved a mean Dice coefficient of "
            f"{d['mean']:.4f} ± {d['std']:.4f} (95% CI: [{d['ci_lower']:.4f}, {d['ci_upper']:.4f}]) "
            f"and IoU of {io['mean']:.4f} ± {io['std']:.4f} on the held-out test set "
            f"(n={len(dataset)} volumes), with sensitivity of {se['mean']:.4f} ± {se['std']:.4f} "
            f"and specificity of {sp['mean']:.4f} ± {sp['std']:.4f}.\n"
        )
        f.write("=" * 70 + "\n")

    print(f"✓ Report        → {report_path}")
    print("\n" + "=" * 70)
    print("PAPER-READY LINE:")
    print(f"Dice {d['mean']:.4f} ± {d['std']:.4f} "
          f"(95% CI [{d['ci_lower']:.4f}, {d['ci_upper']:.4f}]), "
          f"IoU {io['mean']:.4f} ± {io['std']:.4f}, "
          f"Sensitivity {se['mean']:.4f}, Specificity {sp['mean']:.4f}")
    print("=" * 70)

    return results_dict


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained 3D ResNet segmentation model on test set"
    )
    parser.add_argument('--experiment_dir',  required=True,
                        help='Output dir from train_fold_with_cv.py (contains final_model.pth)')
    parser.add_argument('--test_data_file',  required=True,
                        help='Path to test CSV')
    parser.add_argument('--output_dir',      required=True,
                        help='Where to save evaluation outputs')
    parser.add_argument('--modality',        default='t2',
                        choices=['flair', 't1', 't1ce', 't2'])
    parser.add_argument('--model_depth',     type=int, default=34)
    parser.add_argument('--batch_size',      type=int, default=2)
    parser.add_argument('--num_workers',     type=int, default=4)

    args = parser.parse_args()
    run_evaluation(args)