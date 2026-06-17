#!/usr/bin/env python
"""
Utility functions for segmentation training.
"""
import os
import logging
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

def save_model(model, save_path):
    """ Save model weights. """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")


def log_training(epoch, train_loss, val_loss, val_metrics):
    """ Print training progress. """
    print(f"📌 Epoch {epoch+1} | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Accuracy: {val_metrics['accuracy']:.4f} | "
          f"F1: {val_metrics['f1_score']:.4f} | AUC: {val_metrics['roc_auc']:.4f}")

def setup_logging(log_dir, experiment_name):
    """Configure logging settings."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = os.path.join(log_dir, f'{experiment_name}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def dice_coefficient(pred, target, threshold=0.5):
    """
    Calculate Hard Dice coefficient for evaluation with thresholding.
    This version converts the predicted probabilities to binary values using a threshold.
    """
    smooth = 1.0
    
    # Apply sigmoid and threshold for binary prediction
    pred = (torch.sigmoid(pred) > threshold).float()
    
    # Make tensors contiguous before flattening
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def create_cv_splits(dataset, n_folds=5, seed=42):
    """
    Create cross-validation splits based on subject_session.
    
    Args:
        dataset: SegmentationDataset object
        n_folds: Number of CV folds
        seed: Random seed for reproducibility
        
    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    # Extract unique subject_sessions
    subject_sessions = dataset.df['Subject_Session'].unique().tolist()
    np.random.seed(seed)
    np.random.shuffle(subject_sessions)
    
    # Create K-Fold splitter
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Create mapping from subject_session to indices
    subject_session_to_idx = {}
    for i, row in dataset.df.iterrows():
        subject_session = row['Subject_Session']
        if subject_session not in subject_session_to_idx:
            subject_session_to_idx[subject_session] = []
        subject_session_to_idx[subject_session].append(i)
    
    # Create CV splits
    cv_splits = []
    
    for train_idx, val_idx in kf.split(subject_sessions):
        # Get subject_sessions for this split
        train_subjects = [subject_sessions[i] for i in train_idx]
        val_subjects = [subject_sessions[i] for i in val_idx]
        
        # Get all sample indices for train and validation subjects
        train_indices = []
        for subject in train_subjects:
            if subject in subject_session_to_idx:
                train_indices.extend(subject_session_to_idx[subject])
            
        val_indices = []
        for subject in val_subjects:
            if subject in subject_session_to_idx:
                val_indices.extend(subject_session_to_idx[subject])
        
        cv_splits.append((train_indices, val_indices))
    
    return cv_splits
