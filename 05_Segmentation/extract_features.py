#!/usr/bin/env python3
"""
extract_features.py
-------------------
Extracts ResNet34 deep features from T2w MRI volumes using the fine-tuned
tumor segmentation model, for use as imaging features in the DL-M1 survival
pipeline.

Features extracted (via forward hooks on ResNet34 layer3 and layer4):
  - layer3 GAP  (N × 256)  Global Average Pooling
  - layer3 GMP  (N × 256)  Global Max Pooling
  - layer4 GAP  (N × 512)
  - layer4 GMP  (N × 512)
  Plus combined variants (layer3_4_gap_gmp = 1536-dim, used in the paper).

Usage:
    python extract_features.py \\
        --excel_path   /path/to/LGG_Subject_Feature_Extraction.xlsx \\
        --image_dirs   /path/to/MRI_data \\
        --output_dir   features/

    # Use the shipped final model (default):
    python extract_features.py --excel_path subjects.xlsx --image_dirs /mri

    # Specify a different model checkpoint:
    python extract_features.py --excel_path subjects.xlsx \\
        --image_dirs /mri --model_path /path/to/checkpoint.pth

Input Excel columns required:
    SubjectID, Session, Cohort

Image search: the script tries multiple naming conventions and nested folder
structures automatically. Pass multiple --image_dirs to search across them.
"""

import os
import sys
import json
import types
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# All model files live in the same folder as this script — no sys.path hacks needed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CombinedSegmentationDataset
import model as model_utils

# Default model path: final_model/ next to this script
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "final_model" / "final_model.pth"


# ===========================================================================
# LOGGING
# ===========================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"feature_extraction_{ts}.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ===========================================================================
# IMAGE FINDER — searches multiple dirs with multiple naming conventions
# ===========================================================================

T2_PATTERNS = [
    "{ss}_T2_ss_norm.nii.gz",
    "{ss}_t2_ss_norm.nii.gz",
    "{ss}_T2_ss_norm.nii",
    "{ss}_T2.nii.gz",
    "{ss}_t2.nii.gz",
    "{ss}_T2.nii",
    "{ss}.nii.gz",
    "{ss}.nii",
]


def find_image(subject_id: str, session: str, search_dirs: list) -> str | None:
    ss = f"{subject_id}_{session}"
    for base_dir in search_dirs:
        if not os.path.isdir(base_dir):
            continue
        # flat patterns
        for pat in T2_PATTERNS:
            p = os.path.join(base_dir, pat.format(ss=ss))
            if os.path.exists(p):
                return p
        # nested subject/session
        nested = Path(base_dir) / str(subject_id) / str(session)
        if nested.is_dir():
            for pat in T2_PATTERNS:
                p = nested / pat.format(ss=ss)
                if p.exists():
                    return str(p)
            found = list(nested.glob("*T2*.nii.gz")) + list(nested.glob("*.nii.gz"))
            if found:
                return str(found[0])
        # nested subject only
        nested_subj = Path(base_dir) / str(subject_id)
        if nested_subj.is_dir():
            for pat in T2_PATTERNS:
                p = nested_subj / pat.format(ss=ss)
                if p.exists():
                    return str(p)
            found = (list(nested_subj.glob("*T2*.nii.gz")) +
                     list(nested_subj.glob(f"*{session}*.nii.gz")) +
                     list(nested_subj.glob("*.nii.gz")))
            if found:
                return str(found[0])
        # glob fallback
        for gpat in [f"{ss}*T2*.nii.gz", f"{ss}*.nii.gz", f"*{subject_id}*{session}*T2*.nii.gz"]:
            found = list(Path(base_dir).glob(gpat))
            if found:
                return str(found[0])
    return None


# ===========================================================================
# DATASET
# ===========================================================================

class LGGFeatureDataset(Dataset):
    def __init__(self, excel_path: str, search_dirs: list, logger: logging.Logger):
        self.logger      = logger
        self.search_dirs = search_dirs
        self.records     = []
        self.missing     = []

        df = pd.read_excel(excel_path)
        logger.info(f"Excel loaded: {len(df)} subjects  |  columns: {list(df.columns)}")

        for _, row in df.iterrows():
            subject_id = str(row["SubjectID"]).strip()
            session    = str(int(row["Session"])).strip()
            cohort     = str(row["Cohort"]).strip()

            img_path = find_image(subject_id, session, search_dirs)
            if img_path is None:
                self.missing.append(f"{subject_id}_{session}")
            else:
                self.records.append({
                    "subject_session": f"{subject_id}_{session}",
                    "subject_id": subject_id,
                    "session":    session,
                    "cohort":     cohort,
                    "img_path":   img_path,
                })

        logger.info(f"Found: {len(self.records)}  |  Missing: {len(self.missing)}")
        if self.missing:
            logger.warning(f"Missing subjects (first 20): {self.missing[:20]}")
            with open("missing_subjects.txt", "w") as f:
                f.write("\n".join(self.missing))
            logger.info("Full missing list saved → missing_subjects.txt")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = nib.load(rec["img_path"]).get_fdata(dtype=np.float32)
        img = CombinedSegmentationDataset._normalize(img)
        assert img.min() >= 0.0 and img.max() <= 1.0, (
            f"Normalisation out of [0,1] for {rec['subject_session']}: "
            f"min={img.min():.4f}, max={img.max():.4f}"
        )
        return torch.from_numpy(np.expand_dims(img, axis=0)).float(), idx

    def get_record(self, idx):
        return self.records[idx]


# ===========================================================================
# MODEL LOADER
# ===========================================================================

class SegmentationFeatureExtractor(nn.Module):
    """
    Loads the fine-tuned ResNet34 segmentation weights and exposes the
    backbone layers for hook-based feature extraction.

    Attributes:
        base_model  — the unwrapped ResNet34 (layer1..layer4 accessible directly)
    """

    def __init__(self, model_path: str, device: torch.device, model_depth: int = 34):
        super().__init__()

        opt = types.SimpleNamespace(
            model='resnet',
            model_depth=model_depth,
            input_W=128, input_H=128, input_D=128,
            resnet_shortcut='B',
            no_cuda=(device.type == 'cpu'),
            n_seg_classes=2,
            gpu_id=[0],
            phase='test',
            pretrain_path=None,
            new_layer_names=['conv_seg'],
        )

        net, _ = model_utils.generate_model(opt)

        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Strip DataParallel 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        net.module.load_state_dict(state_dict, strict=True)

        # Expose unwrapped backbone so hooks attach to layer3 / layer4 directly
        self.base_model = net.module

    def forward(self, x):
        return self.base_model(x)


# ===========================================================================
# FEATURE EXTRACTOR — forward hooks on layer3 and layer4
# ===========================================================================

class MultiLayerExtractor:
    """
    Registers forward hooks on ResNet34 layer3 and layer4.

    Feature map sizes (ResNet34):
      layer3 → (B, 256, D/4, H/4, W/4)   → GAP/GMP → (B, 256)
      layer4 → (B, 512, D/8, H/8, W/8)   → GAP/GMP → (B, 512)
    """

    def __init__(self, model: SegmentationFeatureExtractor):
        self.model  = model
        self._raw   = {}
        self._hooks = []
        self._register()

    def _register(self):
        def _hook(name):
            def fn(module, inp, out):
                self._raw[name] = out.detach().cpu()
            return fn

        self._hooks.append(self.model.base_model.layer3.register_forward_hook(_hook("layer3")))
        self._hooks.append(self.model.base_model.layer4.register_forward_hook(_hook("layer4")))

    def cleanup(self):
        for h in self._hooks:
            h.remove()

    def extract(self, loader: DataLoader, device: torch.device, logger: logging.Logger) -> dict:
        results = {k: [] for k in ["layer3_gap", "layer3_gmp", "layer4_gap", "layer4_gmp", "indices"]}

        with torch.no_grad():
            for batch_idx, (volumes, indices) in enumerate(loader):
                volumes = volumes.to(device)
                self.model(volumes)

                for layer, dim in [("layer3", 256), ("layer4", 512)]:
                    feat = self._raw[layer]                        # (B, C, D, H, W)
                    gap  = feat.mean(dim=[2, 3, 4]).numpy()        # (B, C)
                    gmp  = feat.amax(dim=[2, 3, 4]).numpy()        # (B, C)
                    results[f"{layer}_gap"].append(gap)
                    results[f"{layer}_gmp"].append(gmp)

                results["indices"].extend(indices.tolist())

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Batch {batch_idx + 1}/{len(loader)}")

        return {k: np.concatenate(v, axis=0) if k != "indices" else v
                for k, v in results.items()}


# ===========================================================================
# OUTPUT
# ===========================================================================

def build_meta_df(dataset: LGGFeatureDataset, indices: list) -> pd.DataFrame:
    return pd.DataFrame([dataset.get_record(i) for i in indices])


def save_csv(meta_df: pd.DataFrame, features: np.ndarray, name: str,
             output_dir: str, timestamp: str, logger: logging.Logger) -> str:
    os.makedirs(output_dir, exist_ok=True)
    feat_cols = [f"{name}_{i}" for i in range(features.shape[1])]
    df = pd.concat([meta_df.reset_index(drop=True),
                    pd.DataFrame(features, columns=feat_cols)], axis=1)
    path = os.path.join(output_dir, f"plgg_features_{name}_{timestamp}.csv")
    df.to_csv(path, index=False)
    logger.info(f"  Saved {name}: {df.shape}  →  {path}")
    return path


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract ResNet34 deep features from T2w MRI for pLGG survival modeling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal — uses the shipped final_model.pth
  python extract_features.py \\
      --excel_path subjects.xlsx \\
      --image_dirs /data/MRI_pLGG

  # Multiple image directories
  python extract_features.py \\
      --excel_path subjects.xlsx \\
      --image_dirs /data/MRI_pLGG/Images /data/clinical_only_t2 \\
      --output_dir features/

  # Custom model checkpoint
  python extract_features.py \\
      --excel_path subjects.xlsx \\
      --image_dirs /data/MRI \\
      --model_path /path/to/my_checkpoint.pth
""",
    )
    parser.add_argument("--excel_path",  required=True,
                        help="Excel file listing subjects (columns: SubjectID, Session, Cohort)")
    parser.add_argument("--image_dirs",  nargs="+", required=True,
                        help="One or more directories to search for T2w NIfTI files")
    parser.add_argument("--model_path",  default=str(DEFAULT_MODEL_PATH),
                        help=f"Path to the segmentation model .pth (default: final_model/final_model.pth)")
    parser.add_argument("--output_dir",  default="features",
                        help="Where to save the output CSV files (default: features/)")
    parser.add_argument("--log_dir",     default="logs/feature_extraction")
    parser.add_argument("--model_depth", type=int, default=34,
                        help="ResNet depth (default: 34)")
    parser.add_argument("--batch_size",  type=int, default=4,
                        help="Batch size — keep 2–4 for 3D volumes (default: 4)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger    = setup_logging(args.log_dir)

    logger.info("=" * 70)
    logger.info("pLGG ResNet34 Feature Extraction")
    logger.info("=" * 70)
    logger.info(f"Excel      : {args.excel_path}")
    logger.info(f"Model      : {args.model_path}")
    logger.info(f"Output dir : {args.output_dir}")
    logger.info(f"Image dirs :")
    for d in args.image_dirs:
        status = "✓" if os.path.isdir(d) else "✗ NOT FOUND"
        logger.info(f"  [{status}]  {d}")

    for path, name in [(args.excel_path, "Excel"), (args.model_path, "Model weights")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Dataset
    dataset = LGGFeatureDataset(args.excel_path, args.image_dirs, logger)
    if len(dataset) == 0:
        raise RuntimeError("No subjects with images found. Check --image_dirs and --excel_path.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        drop_last=False)

    # Model
    logger.info("\nLoading segmentation model...")
    model = SegmentationFeatureExtractor(args.model_path, device, args.model_depth)
    model.to(device)
    model.eval()
    logger.info("Model loaded ✓")

    # Extract
    logger.info("\nExtracting features from layer3 and layer4 (GAP + GMP)...")
    extractor = MultiLayerExtractor(model)
    results   = extractor.extract(loader, device, logger)
    extractor.cleanup()

    l3_gap  = results["layer3_gap"]   # (N, 256)
    l3_gmp  = results["layer3_gmp"]   # (N, 256)
    l4_gap  = results["layer4_gap"]   # (N, 512)
    l4_gmp  = results["layer4_gmp"]   # (N, 512)
    indices = results["indices"]

    logger.info(f"\nExtraction complete: {len(indices)} subjects")
    meta_df = build_meta_df(dataset, indices)

    # Save all pooling variants
    logger.info("\nSaving CSVs...")
    save_csv(meta_df, l3_gap,                                          "layer3_gap",       args.output_dir, timestamp, logger)
    save_csv(meta_df, l3_gmp,                                          "layer3_gmp",       args.output_dir, timestamp, logger)
    save_csv(meta_df, l4_gap,                                          "layer4_gap",       args.output_dir, timestamp, logger)
    save_csv(meta_df, l4_gmp,                                          "layer4_gmp",       args.output_dir, timestamp, logger)
    save_csv(meta_df, np.concatenate([l3_gap, l3_gmp], axis=1),       "layer3_gap_gmp",   args.output_dir, timestamp, logger)
    save_csv(meta_df, np.concatenate([l4_gap, l4_gmp], axis=1),       "layer4_gap_gmp",   args.output_dir, timestamp, logger)
    save_csv(meta_df, np.concatenate([l3_gap, l4_gap], axis=1),       "layer3_4_gap",     args.output_dir, timestamp, logger)
    save_csv(meta_df, np.concatenate([l3_gmp, l4_gmp], axis=1),       "layer3_4_gmp",     args.output_dir, timestamp, logger)
    save_csv(meta_df, np.concatenate([l3_gap, l3_gmp,
                                      l4_gap, l4_gmp], axis=1),       "layer3_4_gap_gmp", args.output_dir, timestamp, logger)

    logger.info(f"\n✓ All features saved to: {args.output_dir}")
    logger.info("\nNext step: pass layer3_4_gap_gmp_*.csv to 01_DLM1_Clinico_ResNet/2_Feature_Engineering.ipynb")


if __name__ == "__main__":
    main()
