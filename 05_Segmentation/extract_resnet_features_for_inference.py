#!/usr/bin/env python3
"""
extract_resnet_features_for_inference.py
-----------------------------------------
One-step script: give it a folder of T2w NIfTI files and it produces
ResNet_Features.xlsx ready to drop straight into

    01_DLM1_Clinico_ResNet/inference.py

No Excel manifest required — SubjectIDs are read directly from the NIfTI
filenames (everything before the first underscore or the full stem).

Feature extracted: layer3 GAP + GMP from the fine-tuned ResNet34 segmentation
backbone → 512-dimensional vector per subject (256 GAP + 256 GMP).
This is the exact feature set used to train the DL-M1 survival model.

Usage (minimal):
    python extract_resnet_features_for_inference.py \\
        --image_dir /path/to/T2w_nifti_files

    Output written to: ResNet_Features.xlsx  (in --output_dir, default: .)

Usage (full):
    python extract_resnet_features_for_inference.py \\
        --image_dir  /path/to/T2w_nifti_files \\
        --model_path final_model/final_model.pth \\
        --output_dir /path/to/results \\
        --batch_size 2

Image naming conventions supported (any of these will be matched):
    C1234567_4762_T2_ss_norm.nii.gz   → SubjectID = C1234567
    sub-001_T2.nii.gz                 → SubjectID = sub-001
    C9876543.nii.gz                   → SubjectID = C9876543
    any_file.nii.gz                   → SubjectID = any_file (full stem)

After running, pass the output file to inference.py:
    cd ../01_DLM1_Clinico_ResNet
    python inference.py \\
        --input-dir /path/to/results \\
        --output-dir /path/to/results
"""

import os
import sys
import re
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CombinedSegmentationDataset
import model as model_utils

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "final_model" / "final_model.pth"

# ── Filename → SubjectID ──────────────────────────────────────────────────────

def subject_id_from_filename(path: Path) -> str:
    """
    Extract SubjectID from a NIfTI filename.
    Matches the leading C<digits> or sub<digits> token if present,
    otherwise uses the full file stem (minus .nii or .nii.gz).
    """
    stem = path.name
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    m = re.match(r"^(C\d+|sub[-_]?\d+)", stem)
    return m.group(1) if m else stem.split("_")[0]


# ── Dataset ───────────────────────────────────────────────────────────────────

class NIfTIFolderDataset(Dataset):
    """
    Scans image_dir for all *.nii.gz / *.nii files, normalises each volume
    to [0, 1] using the same robust percentile clipping applied during
    segmentation training, and yields (tensor, index) pairs.
    """

    EXTENSIONS = (".nii.gz", ".nii")

    def __init__(self, image_dir: str, logger: logging.Logger):
        self.logger  = logger
        self.records = []

        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")

        candidates = sorted(
            p for p in image_dir.iterdir()
            if any(str(p).endswith(ext) for ext in self.EXTENSIONS)
        )

        if not candidates:
            raise RuntimeError(
                f"No .nii.gz / .nii files found in: {image_dir}\n"
                "Check --image_dir points to the folder containing your T2w volumes."
            )

        for path in candidates:
            subject_id = subject_id_from_filename(path)
            self.records.append({"subject_id": subject_id, "img_path": str(path)})
            logger.info(f"  Found: {path.name}  →  SubjectID = {subject_id}")

        logger.info(f"\n{len(self.records)} volumes found in {image_dir}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path = self.records[idx]["img_path"]
        img  = nib.load(path).get_fdata(dtype=np.float32)
        if img.ndim > 3:
            img = img[..., 0]
        img = CombinedSegmentationDataset._normalize(img)
        return torch.from_numpy(np.expand_dims(img, 0)).float(), idx

    def subject_id(self, idx: int) -> str:
        return self.records[idx]["subject_id"]


# ── Model ─────────────────────────────────────────────────────────────────────

class SegmentationFeatureExtractor(nn.Module):
    """Loads fine-tuned ResNet34 weights and exposes the backbone for hooks."""

    def __init__(self, model_path: str, device: torch.device, model_depth: int = 34):
        super().__init__()
        opt = types.SimpleNamespace(
            model="resnet", model_depth=model_depth,
            input_W=128, input_H=128, input_D=128,
            resnet_shortcut="B",
            no_cuda=(device.type == "cpu"),
            n_seg_classes=1, gpu_id=[0],
            phase="test", pretrain_path=None,
            new_layer_names=["conv_seg"],
        )
        net, _ = model_utils.generate_model(opt)
        ckpt = torch.load(model_path, map_location=device)
        sd   = ckpt.get("model_state_dict", ckpt)
        sd   = {k.replace("module.", ""): v for k, v in sd.items()}
        net.module.load_state_dict(sd, strict=True)
        self.base_model = net.module  # unwrapped ResNet34

    def forward(self, x):
        return self.base_model(x)


# ── Feature extraction — layer3 GAP + GMP (512-dim) ──────────────────────────
# layer3 GAP (256) concatenated with layer3 GMP (256) = 512 features total.
# This is the exact feature set used to train the DL-M1 survival model.

def extract_layer3_gap_gmp(model: SegmentationFeatureExtractor,
                            loader: DataLoader,
                            device: torch.device,
                            logger: logging.Logger) -> tuple[np.ndarray, list[int]]:
    """
    Returns (features, indices):
        features — (N, 512) float32  [layer3_GAP (256) | layer3_GMP (256)]
        indices  — dataset indices in extraction order
    """
    collected: list[np.ndarray] = []
    all_indices: list[int] = []
    captured: dict = {}

    def _hook(module, inp, out):
        captured["layer3"] = out.detach().cpu()

    handle = model.base_model.layer3.register_forward_hook(_hook)

    try:
        with torch.no_grad():
            for batch_idx, (volumes, indices) in enumerate(loader):
                model(volumes.to(device))
                feat = captured["layer3"]                      # (B, 256, D, H, W)
                gap  = feat.mean(dim=[2, 3, 4]).numpy()        # (B, 256)
                gmp  = feat.amax(dim=[2, 3, 4]).numpy()        # (B, 256)
                collected.append(np.concatenate([gap, gmp], axis=1))  # (B, 512)
                all_indices.extend(indices.tolist())
                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
                    logger.info(f"  Processed {batch_idx + 1}/{len(loader)} batches")
    finally:
        handle.remove()

    return np.concatenate(collected, axis=0), all_indices


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract ResNet34 features from T2w MRI → ResNet_Features.xlsx for DL-M1 inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal — point at a folder of T2w NIfTI files
  python extract_resnet_features_for_inference.py --image_dir /data/T2w_scans

  # Custom output location
  python extract_resnet_features_for_inference.py \\
      --image_dir /data/T2w_scans \\
      --output_dir /results/my_cohort

  # Then run DL-M1 inference:
  cd ../01_DLM1_Clinico_ResNet
  python inference.py --input-dir /results/my_cohort --output-dir /results/my_cohort
""",
    )
    parser.add_argument(
        "--image_dir", required=True,
        help="Folder containing T2w NIfTI files (.nii.gz or .nii). "
             "SubjectIDs are parsed automatically from filenames.",
    )
    parser.add_argument(
        "--model_path", default=str(DEFAULT_MODEL_PATH),
        help="Path to the segmentation model checkpoint "
             "(default: final_model/final_model.pth next to this script)",
    )
    parser.add_argument(
        "--output_dir", default=".",
        help="Where to write ResNet_Features.xlsx (default: current directory)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size — keep 1–4 for 3D volumes (default: 2)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader workers (default: 0 — safe on all platforms)",
    )
    parser.add_argument(
        "--model_depth", type=int, default=34,
        help="ResNet depth — must match the checkpoint (default: 34)",
    )
    args = parser.parse_args()

    # Logging
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"feature_extraction_{ts}.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 65)
    logger.info("pLGG ResNet34 Feature Extraction — DL-M1 Inference Prep")
    logger.info("=" * 65)
    logger.info(f"Image directory : {args.image_dir}")
    logger.info(f"Model weights   : {args.model_path}")
    logger.info(f"Output          : {args.output_dir}/ResNet_Features.xlsx")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model weights not found: {args.model_path}\n"
            "If you cloned the repo with Git LFS disabled, run:\n"
            "    git lfs pull\n"
            "to download final_model/final_model.pth."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device          : {device}\n")

    # Dataset & loader
    logger.info("Scanning image directory...")
    dataset = NIfTIFolderDataset(args.image_dir, logger)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Load model
    logger.info("\nLoading segmentation model...")
    model = SegmentationFeatureExtractor(args.model_path, device, args.model_depth)
    model.to(device)
    model.eval()
    logger.info("Model loaded ✓")

    # Extract
    logger.info(f"\nExtracting layer3 GAP+GMP features ({len(dataset)} subjects)...")
    features, indices = extract_layer3_gap_gmp(model, loader, device, logger)
    logger.info(f"Extraction complete: {features.shape}  (subjects × 512 features [layer3 GAP|GMP])")

    # Build output DataFrame
    subject_ids = [dataset.subject_id(i) for i in indices]
    feat_cols   = [f"feature_{i:03d}" for i in range(features.shape[1])]
    df_out      = pd.DataFrame(features, columns=feat_cols)
    df_out.insert(0, "SubjectID", subject_ids)

    # Save
    out_path = Path(args.output_dir) / "ResNet_Features.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(str(out_path), index=False)

    logger.info(f"\n✓ Saved: {out_path}")
    logger.info(f"  Subjects : {len(df_out)}")
    logger.info(f"  Features : {features.shape[1]} (layer3 GAP+GMP)")

    logger.info("\n" + "=" * 65)
    logger.info("Next step — run DL-M1 risk inference:")
    logger.info("")
    logger.info("  cd ../01_DLM1_Clinico_ResNet")
    logger.info(f"  python inference.py \\")
    logger.info(f"      --input-dir  {Path(args.output_dir).resolve()} \\")
    logger.info(f"      --output-dir {Path(args.output_dir).resolve()}")
    logger.info("")
    logger.info("  Make sure Clinical_Features.xlsx is also in that folder.")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
