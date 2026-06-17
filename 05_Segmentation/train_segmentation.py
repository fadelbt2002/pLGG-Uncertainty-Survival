# save as train_fold_with_cv.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.cuda.amp import autocast, GradScaler
from itertools import product
from scipy import stats

from dataset import CombinedSegmentationDataset
from utils import setup_logging, dice_coefficient
import model as model_utils


# ===========================================================================
# LOSS: BCE + Soft Dice
# ===========================================================================

class BCEDiceLoss(nn.Module):
    """
    Combined BCE (on logits) + Soft Dice (on sigmoid probabilities).
    Weighted equally by default.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.smooth      = smooth
        self.bce         = nn.BCEWithLogitsLoss()

    def _soft_dice(self, probs, target):
        p = probs.contiguous().view(-1)
        t = target.contiguous().view(-1)
        intersection = (p * t).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)

    def forward(self, logits, target):
        return (self.bce_weight  * self.bce(logits, target) +
                self.dice_weight * self._soft_dice(torch.sigmoid(logits), target))


# ===========================================================================
# HELPERS
# ===========================================================================

def calculate_95_ci(values):
    n     = len(values)
    mean  = np.mean(values)
    std   = np.std(values, ddof=1)
    sem   = std / np.sqrt(n)
    t_val = stats.t.ppf(0.975, n - 1)
    return mean, mean - t_val * sem, mean + t_val * sem, std


def build_model(args, input_size, device):
    class _Opt:
        model           = 'resnet'
        resnet_shortcut = 'B'
        no_cuda         = False
        gpu_id          = [0]
        phase           = 'train'
        n_seg_classes   = 1
        new_layer_names = ['conv_seg']

        def __init__(self, args, input_size):
            self.model_depth   = args.model_depth
            self.input_D       = input_size[0]
            self.input_H       = input_size[1]
            self.input_W       = input_size[2]
            self.pretrain_path = args.pretrained_path

    m, _ = model_utils.generate_model(_Opt(args, input_size))
    return m.to(device)


def build_scheduler(optimizer, target_lr, total_epochs, warmup_epochs=5, min_lr=1e-6):
    """
    Linear warmup (1e-6 -> target_lr) over warmup_epochs, then cosine
    decay (target_lr -> min_lr) over the remaining epochs.
    Scheduler steps once per epoch (call scheduler.step() after each epoch).
    """
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=min_lr / target_lr,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
        eta_min=min_lr,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


# ===========================================================================
# SINGLE FOLD TRAINING
# ===========================================================================

def train_single_fold(fold_idx, train_indices, val_indices,
                      dataset, hyperparams, args, device, logger, input_size):
    """
    Train for one CV fold.
    Early stopping monitors *validation loss* (minimize).
    Returns (best_val_loss, best_val_dice, best_epoch).
    """
    batch_size, lr, weight_decay, acc_steps = hyperparams
    logger.info(f"  Fold {fold_idx+1}: {len(train_indices)} train / {len(val_indices)} val")

    train_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=args.num_workers, pin_memory=True,
        drop_last=False, persistent_workers=args.num_workers > 0,
    )

    model     = build_model(args, input_size, device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, lr, args.num_epochs)
    criterion = BCEDiceLoss()
    # FIX 1: Gate AMP + GradScaler to CUDA only — autocast/GradScaler are
    # no-ops on CPU but GradScaler(enabled=False) is safer and explicit.
    amp    = torch.cuda.is_available()
    scaler = GradScaler(enabled=amp)

    best_val_loss = float('inf')
    best_val_dice = -1.0
    best_epoch    = 0
    no_improve    = 0

    for epoch in range(args.num_epochs):
        # ---- Train ----
        model.train()
        t_loss = t_dice = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for bi, (images, masks, _) in enumerate(train_loader):
            try:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)

                with autocast(enabled=amp):
                    out = model(images)
                    if out.shape != masks.shape:
                        out = nn.functional.interpolate(
                            out, size=masks.shape[2:], mode='trilinear', align_corners=False)
                    loss = criterion(out, masks)
                    if acc_steps > 1:
                        loss = loss / acc_steps

                scaler.scale(loss).backward()

                if (bi + 1) % acc_steps == 0 or (bi + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                with torch.no_grad():
                    t_loss    += loss.item() * (acc_steps if acc_steps > 1 else 1)
                    t_dice    += dice_coefficient(out, masks).item()
                    n_batches += 1

            except RuntimeError as e:
                logger.error(f"Train batch {bi} error: {e}")
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        # ---- Validate ----
        model.eval()
        v_loss = v_dice = 0.0
        n_val = 0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                try:
                    images = images.to(device, non_blocking=True)
                    masks  = masks.to(device, non_blocking=True)
                    out    = model(images)
                    if out.shape != masks.shape:
                        out = nn.functional.interpolate(
                            out, size=masks.shape[2:], mode='trilinear', align_corners=False)
                    v_loss += criterion(out, masks).item()
                    v_dice += dice_coefficient(out, masks).item()
                    n_val  += 1
                except RuntimeError as e:
                    logger.error(f"Val batch error: {e}")

        avg_tl = t_loss / max(n_batches, 1)
        avg_td = t_dice / max(n_batches, 1)
        avg_vl = v_loss / max(n_val, 1)
        avg_vd = v_dice / max(n_val, 1)

        scheduler.step()
        logger.info(
            f"    Epoch {epoch+1:3d} | "
            f"train loss={avg_tl:.4f} dice={avg_td:.4f} | "
            f"val loss={avg_vl:.4f} dice={avg_vd:.4f}"
        )

        # Early stopping on val loss
        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            best_val_dice = avg_vd
            best_epoch    = epoch + 1
            no_improve    = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            logger.info(f"    Early stopping at epoch {epoch+1}")
            break

    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return best_val_loss, best_val_dice, best_epoch


# ===========================================================================
# MAIN
# ===========================================================================

def train_with_cv_hp_search(args):
    """
    Non-nested 5-fold CV hyperparameter search, then train final model on
    entire training set with the best hyperparameters.
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir, f"cv_hp_search_{timestamp}")
    logger.info(f"Arguments: {args}")

    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # ------------------------------------------------------------------
    # Dataset — robust percentile normalisation baked in via subclass
    # ------------------------------------------------------------------
    dataset = CombinedSegmentationDataset(args.data_file)
    logger.info(f"Dataset: {len(dataset)} samples")

    if args.modality:
        dataset = dataset.filter_by_modality(args.modality)
        logger.info(f"After modality filter: {len(dataset)} samples")

    sample_img, _, _ = dataset[0]
    input_size = sample_img.shape[1:]   # (D, H, W)
    logger.info(f"Input size: {input_size}")

    # ------------------------------------------------------------------
    # 5-fold CV splits (subject-session stratified)
    # ------------------------------------------------------------------
    # FIX 2: create_cv_splits is called AFTER filter_by_modality so it
    # receives the already-filtered dataset. We then assert that every
    # index it emits is within bounds and that train/val never overlap.
    cv_splits = CombinedSegmentationDataset.create_cv_splits(
        dataset, n_folds=args.n_folds, seed=args.seed)

    logger.info(f"\n{args.n_folds}-fold CV splits:")
    for i, (tr, va) in enumerate(cv_splits):
        assert max(max(tr), max(va)) < len(dataset), (
            f"Fold {i+1}: index out of range — "
            f"max index {max(max(tr), max(va))} >= dataset size {len(dataset)}. "
            f"create_cv_splits() was likely called on the pre-filter dataset."
        )
        assert set(tr).isdisjoint(set(va)), (
            f"Fold {i+1}: train/val overlap detected — "
            f"{len(set(tr) & set(va))} shared indices."
        )
        logger.info(f"  Fold {i+1}: {len(tr)} train / {len(va)} val  ✓ indices OK")

    splits_path = os.path.join(output_dir, "cv_splits.json")
    with open(splits_path, 'w') as f:
        json.dump([
            {'fold': i + 1, 'train': tr.tolist() if hasattr(tr, 'tolist') else list(tr),
                             'val':   va.tolist() if hasattr(va, 'tolist') else list(va)}
            for i, (tr, va) in enumerate(cv_splits)
        ], f, indent=2)
    logger.info(f"CV splits saved → {splits_path}")

    # ------------------------------------------------------------------
    # Hyperparameter grid search
    # ------------------------------------------------------------------
    hp_grid = list(product(
        args.batch_sizes, args.learning_rates,
        args.weight_decays, args.accumulation_steps,
    ))
    logger.info(f"\nGrid: {len(hp_grid)} combinations × {args.n_folds} folds")

    hp_results = []

    for hp_idx, hp in enumerate(hp_grid):
        bs, lr, wd, acc = hp
        logger.info(f"\n{'='*80}")
        logger.info(f"HP {hp_idx+1}/{len(hp_grid)}: bs={bs}, lr={lr}, wd={wd}, acc={acc}")
        logger.info(f"{'='*80}")

        fold_losses, fold_dices, fold_epochs = [], [], []

        for fold_idx, (tr_idx, va_idx) in enumerate(cv_splits):
            logger.info(f"\n  Fold {fold_idx+1}/{args.n_folds}")
            vl, vd, ep = train_single_fold(
                fold_idx, tr_idx, va_idx,
                dataset, hp, args, device, logger, input_size)
            fold_losses.append(vl)
            fold_dices.append(vd)
            fold_epochs.append(ep)
            logger.info(f"  → val loss={vl:.4f}, dice={vd:.4f}, best epoch={ep}")

        mean_loss, ci_lo_l, ci_hi_l, std_loss = calculate_95_ci(fold_losses)
        mean_dice, ci_lo_d, ci_hi_d, std_dice = calculate_95_ci(fold_dices)
        mean_ep = float(np.mean(fold_epochs))

        logger.info(f"\n  CV val loss : {mean_loss:.4f} ± {std_loss:.4f} [{ci_lo_l:.4f}, {ci_hi_l:.4f}]")
        logger.info(f"  CV val dice : {mean_dice:.4f} ± {std_dice:.4f} [{ci_lo_d:.4f}, {ci_hi_d:.4f}]")
        logger.info(f"  Mean best epoch: {mean_ep:.1f}")

        hp_results.append({
            'hp_idx': hp_idx, 'batch_size': bs, 'learning_rate': lr,
            'weight_decay': wd, 'accumulation_steps': acc,
            'fold_val_losses':  [float(v) for v in fold_losses],
            'mean_val_loss':    float(mean_loss), 'std_val_loss':  float(std_loss),
            'ci_lower_loss':    float(ci_lo_l),   'ci_upper_loss': float(ci_hi_l),
            'fold_val_dices':   [float(v) for v in fold_dices],
            'mean_val_dice':    float(mean_dice),  'std_val_dice':  float(std_dice),
            'ci_lower_dice':    float(ci_lo_d),    'ci_upper_dice': float(ci_hi_d),
            'mean_best_epoch':  mean_ep,
            'individual_best_epochs': [int(e) for e in fold_epochs],
        })

    # Select best HP (lowest mean val loss)
    hp_results.sort(key=lambda x: x['mean_val_loss'])
    best_hp = hp_results[0]

    logger.info("\n" + "=" * 80)
    logger.info("BEST HYPERPARAMETERS:")
    logger.info(f"  bs={best_hp['batch_size']}, lr={best_hp['learning_rate']}, "
                f"wd={best_hp['weight_decay']}, acc={best_hp['accumulation_steps']}")
    logger.info(f"  Mean val loss : {best_hp['mean_val_loss']:.4f} ± {best_hp['std_val_loss']:.4f}")
    logger.info(f"  Mean val dice : {best_hp['mean_val_dice']:.4f} ± {best_hp['std_val_dice']:.4f}")
    logger.info("=" * 80)

    with open(os.path.join(output_dir, "hp_results.json"), 'w') as f:
        json.dump(hp_results, f, indent=2)

    # ------------------------------------------------------------------
    # Final model — entire training set, best HPs, no early stopping
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING FINAL MODEL ON ENTIRE TRAINING SET")
    logger.info("=" * 80)

    final_epochs = max(int(best_hp['mean_best_epoch'] * 1.5), 30)
    logger.info(f"Epochs: {final_epochs} (1.5 × CV mean {best_hp['mean_best_epoch']:.1f}, floor=30)")

    final_loader = DataLoader(
        dataset, batch_size=best_hp['batch_size'],
        sampler=SubsetRandomSampler(list(range(len(dataset)))),
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=args.num_workers > 0,
    )

    final_model     = build_model(args, input_size, device)
    final_optimizer = optim.AdamW(
        final_model.parameters(),
        lr=best_hp['learning_rate'], weight_decay=best_hp['weight_decay'])
    final_scheduler = build_scheduler(final_optimizer, best_hp['learning_rate'], final_epochs)
    final_criterion = BCEDiceLoss()
    # FIX 1 (final train): same AMP gating as fold training
    amp          = torch.cuda.is_available()
    final_scaler = GradScaler(enabled=amp)
    acc          = best_hp['accumulation_steps']

    best_train_loss  = float('inf')
    best_final_state = None

    for epoch in range(final_epochs):
        final_model.train()
        t_loss = t_dice = 0.0
        n_batches = 0
        final_optimizer.zero_grad()

        for bi, (images, masks, _) in enumerate(final_loader):
            try:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)

                with autocast(enabled=amp):
                    out = final_model(images)
                    if out.shape != masks.shape:
                        out = nn.functional.interpolate(
                            out, size=masks.shape[2:], mode='trilinear', align_corners=False)
                    loss = final_criterion(out, masks)
                    if acc > 1:
                        loss = loss / acc

                final_scaler.scale(loss).backward()

                if (bi + 1) % acc == 0 or (bi + 1) == len(final_loader):
                    final_scaler.step(final_optimizer)
                    final_scaler.update()
                    final_optimizer.zero_grad()

                with torch.no_grad():
                    t_loss    += loss.item() * (acc if acc > 1 else 1)
                    t_dice    += dice_coefficient(out, masks).item()
                    n_batches += 1

            except RuntimeError as e:
                logger.error(f"Final train batch {bi} error: {e}")
                final_optimizer.zero_grad()
                torch.cuda.empty_cache()

        avg_tl = t_loss / max(n_batches, 1)
        avg_td = t_dice / max(n_batches, 1)
        final_scheduler.step()
        logger.info(f"Epoch {epoch+1:3d}/{final_epochs} | loss={avg_tl:.4f}, dice={avg_td:.4f}")

        if avg_tl < best_train_loss:
            best_train_loss  = avg_tl
            best_final_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
            logger.info(f"  ✓ New best train loss: {best_train_loss:.4f}")

    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save({
        'model_state_dict': best_final_state,
        'hyperparams': {
            'batch_size':        best_hp['batch_size'],
            'learning_rate':     best_hp['learning_rate'],
            'weight_decay':      best_hp['weight_decay'],
            'accumulation_steps': best_hp['accumulation_steps'],
            'model_depth':       args.model_depth,
            'modality':          args.modality,
        },
        'training_samples': len(dataset),
        'best_train_loss':  best_train_loss,
        'final_epochs':     final_epochs,
    }, final_model_path)
    logger.info(f"\n✓ FINAL MODEL SAVED: {final_model_path}")

    del final_model, final_optimizer, final_scheduler, final_scaler
    torch.cuda.empty_cache()

    summary = {
        'n_folds': args.n_folds, 'modality': args.modality,
        'best_hyperparams': {
            'batch_size':        best_hp['batch_size'],
            'learning_rate':     best_hp['learning_rate'],
            'weight_decay':      best_hp['weight_decay'],
            'accumulation_steps': best_hp['accumulation_steps'],
        },
        'cv_performance': {
            'mean_val_loss':   best_hp['mean_val_loss'], 'std_val_loss':  best_hp['std_val_loss'],
            'ci_lower_loss':   best_hp['ci_lower_loss'], 'ci_upper_loss': best_hp['ci_upper_loss'],
            'mean_val_dice':   best_hp['mean_val_dice'], 'std_val_dice':  best_hp['std_val_dice'],
            'fold_val_losses': best_hp['fold_val_losses'],
            'fold_val_dices':  best_hp['fold_val_dices'],
        },
        'mean_best_epoch':     best_hp['mean_best_epoch'],
        'final_model_path':    final_model_path,
        'final_model_epochs':  final_epochs,
        'total_train_samples': len(dataset),
    }
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nAll results saved to {output_dir}")
    return summary


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='5-fold CV HP search + final model training on full training set'
    )
    parser.add_argument('--data_file',       required=True)
    parser.add_argument('--output_dir',      required=True)
    parser.add_argument('--modality',        default='t2',
                        choices=['flair', 't1', 't1ce', 't2'])
    parser.add_argument('--model_depth',     type=int, default=34,
                        choices=[10, 18, 34, 50, 101, 152])
    parser.add_argument('--pretrained_path', default=None)
    parser.add_argument('--num_epochs',      type=int, default=100)
    parser.add_argument('--patience',        type=int, default=7,
                        help='Early stopping patience (val loss, CV folds only)')
    parser.add_argument('--num_workers',     type=int, default=8)
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--n_folds',         type=int, default=5)
    parser.add_argument('--batch_sizes',        type=int,   nargs='+', default=[2, 4])
    parser.add_argument('--learning_rates',     type=float, nargs='+', default=[1e-3, 1e-4, 3e-4])
    parser.add_argument('--weight_decays',      type=float, nargs='+', default=[1e-4, 1e-2])
    parser.add_argument('--accumulation_steps', type=int,   nargs='+', default=[1, 2, 4, 8])

    args = parser.parse_args()
    train_with_cv_hp_search(args)