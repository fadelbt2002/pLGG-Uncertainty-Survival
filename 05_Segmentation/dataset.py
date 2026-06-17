import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
from sklearn.model_selection import KFold


class CombinedSegmentationDataset(Dataset):
    """
    Dataset for brain tumor segmentation supporting both original and
    nnUNet-formatted data.

    Normalisation: robust percentile clipping (1st–99th) followed by
    min-max scaling to [0, 1]. This replaces the previous simple max-
    normalisation, which was vulnerable to MRI outlier voxels.

    CV splits: use create_cv_splits() — stratified by Subject_Session so
    all modalities for a given session stay in the same fold.
    """

    def __init__(self, data_file, transform=None):
        self.data_file = data_file
        self.df        = pd.read_csv(data_file)
        self.transform = transform

        print(f"Dataset initialized with {len(self.df)} samples")
        print(f"Modalities:   {self.df['Modality'].value_counts().to_dict()}")
        print(f"Data sources: {self.df['DataSource'].value_counts().to_dict()}")

    def __len__(self):
        return len(self.df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_volume(path):
        """Load a NIfTI file and return a float32 3-D numpy array."""
        data = nib.load(path).get_fdata()
        if data.ndim > 3:
            data = data[..., 0]
        return data.astype(np.float32)

    @staticmethod
    def _normalize(img):
        """
        Per-volume min-max normalization to [0, 1].
        """
        v_min, v_max = img.min(), img.max()
        if v_max > v_min:
            return (img - v_min) / (v_max - v_min)
        return np.zeros_like(img)  # flat volume

    @staticmethod
    def _binarize_mask(mask):
        """Convert any non-zero label to 1 (tumor vs background)."""
        return (mask > 0).astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        row         = self.df.iloc[idx]
        data_source = row['DataSource']

        img  = self._normalize(self._load_volume(row['ImagePath']))
        mask = self._binarize_mask(self._load_volume(row['LabelPath']))

        img_tensor  = torch.from_numpy(img).unsqueeze(0)   # (1, D, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, D, H, W)

        if self.transform:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        metadata = {
            'subject':     row['Subject'],
            'session':     row['Session'],
            'modality':    row['Modality'],
            'data_source': data_source,
            'image_path':  row['ImagePath'],
            'mask_path':   row['LabelPath'],
        }
        return img_tensor, mask_tensor, metadata

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_modality(self, modality):
        """
        Return a new dataset restricted to one modality.
        Resets the DataFrame index to prevent out-of-bounds errors.
        """
        filtered = self.df[self.df['Modality'] == modality]
        if filtered.empty:
            raise ValueError(f"No samples found for modality '{modality}'")

        new_ds     = CombinedSegmentationDataset(self.data_file, self.transform)
        new_ds.df  = filtered.reset_index(drop=True)

        print(f"Filtered to '{modality}': {len(new_ds)} samples "
              f"from {new_ds.df['Subject_Session'].nunique()} subject-sessions")
        print(f"Data sources: {new_ds.df['DataSource'].value_counts().to_dict()}")
        return new_ds

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    @staticmethod
    def create_cv_splits(dataset, n_folds=5, seed=42):
        """
        Subject-session-stratified K-Fold splits (default 80/20).
        Keeps all modalities for a session together to prevent leakage.

        Returns: list of (train_indices, val_indices) — one tuple per fold.
        """
        subject_sessions = dataset.df['Subject_Session'].unique()
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        splits = []
        for tr_ss_idx, va_ss_idx in kf.split(subject_sessions):
            tr_ss = subject_sessions[tr_ss_idx]
            va_ss = subject_sessions[va_ss_idx]
            tr_idx = dataset.df[dataset.df['Subject_Session'].isin(tr_ss)].index.tolist()
            va_idx = dataset.df[dataset.df['Subject_Session'].isin(va_ss)].index.tolist()
            splits.append((tr_idx, va_idx))

        return splits