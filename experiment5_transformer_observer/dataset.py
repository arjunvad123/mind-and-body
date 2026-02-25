"""
HDF5 Dataset Loader

Memory-mapped access to extracted GPT-2 activations with:
- Sliding window extraction for causal prediction
- Per-layer z-score normalization (critical — different layers have different scales)
- Train/val/test split
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from . import config


class ActivationDataset(Dataset):
    """Dataset of GPT-2 residual stream activations for observer training.

    Each item is a window of consecutive token positions from a single sequence.
    Input: residual stream at all 13 checkpoints for positions 0..T-1
    Target: final-layer residual at position T (the next position)

    Per-layer z-score normalization ensures that layers with different scales
    contribute equally to the observer's input.
    """

    def __init__(
        self,
        h5_path: str,
        split: str = 'train',
        seq_len: int = None,
        normalize: bool = True,
        precompute_stats: bool = True,
    ):
        """
        Args:
            h5_path: Path to HDF5 file with extracted activations
            split: 'train', 'val', or 'test'
            seq_len: Max sequence length for windows (default: config.OBSERVER_MAX_SEQ_LEN)
            normalize: Whether to apply per-layer z-score normalization
            precompute_stats: Whether to compute normalization stats from training split
        """
        self.h5_path = Path(h5_path)
        self.split = split
        self.seq_len = seq_len or config.OBSERVER_MAX_SEQ_LEN
        self.normalize = normalize

        # Open HDF5 to read metadata (keep handle open for fast repeated access)
        self._h5_file = h5py.File(self.h5_path, 'r')
        self.n_sequences = self._h5_file.attrs['n_sequences']
        self.token_seq_len = self._h5_file.attrs['seq_len']
        self.n_checkpoints = self._h5_file.attrs['n_checkpoints']
        self.d_model = self._h5_file.attrs['d_model']
        self._resid_ds = self._h5_file['residual_stream']

        # Compute split indices
        n_train = int(self.n_sequences * config.TRAIN_RATIO)
        n_val = int(self.n_sequences * config.VAL_RATIO)

        if split == 'train':
            self.start_idx = 0
            self.end_idx = n_train
        elif split == 'val':
            self.start_idx = n_train
            self.end_idx = n_train + n_val
        elif split == 'test':
            self.start_idx = n_train + n_val
            self.end_idx = self.n_sequences
        else:
            raise ValueError(f"Unknown split: {split}")

        self.n_items = self.end_idx - self.start_idx

        # Each sequence of length L gives us L-1 training examples
        # (predicting position t+1 from positions 0..t)
        # But we use fixed windows, so each sequence gives ~1 window
        # since seq_len matches token_seq_len in our case

        # Normalization statistics (per-layer mean and std)
        self.layer_means = None
        self.layer_stds = None

        if normalize and precompute_stats:
            self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute per-layer z-score statistics from the training split.

        Each layer of the residual stream has a different scale.
        Normalizing ensures the observer treats all layers equally.
        """
        print(f"Computing normalization statistics from training split...")

        # Sample a subset to compute stats efficiently
        n_sample = min(200, self.n_items)
        sample_indices = np.random.RandomState(42).choice(
            self.n_items, n_sample, replace=False
        ) + self.start_idx

        means = np.zeros((self.n_checkpoints, self.d_model), dtype=np.float64)
        sq_means = np.zeros((self.n_checkpoints, self.d_model), dtype=np.float64)
        count = 0

        for idx in sample_indices:
            # (n_checkpoints, seq_len, d_model)
            data = self._resid_ds[idx]
            # Average over token positions
            layer_mean = data.mean(axis=1)  # (n_checkpoints, d_model)
            layer_sq = (data ** 2).mean(axis=1)
            means += layer_mean
            sq_means += layer_sq
            count += 1

        means /= count
        sq_means /= count
        stds = np.sqrt(np.maximum(sq_means - means ** 2, 1e-8))

        self.layer_means = torch.tensor(means, dtype=torch.float32)  # (n_checkpoints, d_model)
        self.layer_stds = torch.tensor(stds, dtype=torch.float32)

        print(f"  Layer mean norms: {np.linalg.norm(means, axis=1).round(2)}")
        print(f"  Layer std norms:  {np.linalg.norm(stds.astype(np.float32), axis=1).round(2)}")

    def __del__(self):
        """Close HDF5 file handle."""
        if hasattr(self, '_h5_file') and self._h5_file:
            try:
                self._h5_file.close()
            except Exception:
                pass

    def set_normalization_stats(self, means, stds):
        """Set normalization stats externally (for val/test using train stats)."""
        self.layer_means = means
        self.layer_stds = stds

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        """Return a single training example.

        Returns:
            input_resid: (n_checkpoints, seq_len-1, d_model) — residual stream at all layers
            target_resid: (seq_len-1, d_model) — final-layer residual at next position
        """
        actual_idx = self.start_idx + idx

        # (n_checkpoints, seq_len, d_model)
        resid = torch.tensor(self._resid_ds[actual_idx], dtype=torch.float32)

        # Normalize per layer
        if self.normalize and self.layer_means is not None:
            # layer_means: (n_checkpoints, d_model) -> (n_checkpoints, 1, d_model)
            resid = (resid - self.layer_means.unsqueeze(1)) / self.layer_stds.unsqueeze(1)

        # Input: all checkpoints at positions 0..T-2
        # Target: final checkpoint (post last layer) at positions 1..T-1
        input_resid = resid[:, :-1, :]      # (n_checkpoints, seq_len-1, d_model)
        target_resid = resid[-1, 1:, :]     # (seq_len-1, d_model) — final layer, shifted by 1

        return input_resid, target_resid


class ShuffledActivationDataset(Dataset):
    """Control dataset: temporally shuffled activations.

    Same activation statistics, but temporal structure destroyed.
    If the observer trained on this matches the real observer,
    temporal dynamics don't matter.
    """

    def __init__(self, base_dataset: ActivationDataset, seed: int = 42):
        self.base = base_dataset
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        input_resid, target_resid = self.base[idx]

        # Shuffle the temporal dimension (dim=1 of input, dim=0 of target)
        seq_len = input_resid.shape[1]
        perm = torch.tensor(self.rng.permutation(seq_len))

        input_shuffled = input_resid[:, perm, :]
        target_shuffled = target_resid[perm, :]

        return input_shuffled, target_shuffled


def create_dataloaders(
    h5_path: str = None,
    batch_size: int = None,
    num_workers: int = 0,
) -> dict:
    """Create train/val/test dataloaders with shared normalization stats.

    Returns:
        dict with 'train', 'val', 'test' DataLoaders and 'train_dataset' for reference
    """
    h5_path = h5_path or str(config.ACTIVATIONS_PATH)
    batch_size = batch_size or config.BATCH_SIZE

    # Create datasets
    train_ds = ActivationDataset(h5_path, split='train', normalize=True)
    val_ds = ActivationDataset(h5_path, split='val', normalize=True, precompute_stats=False)
    test_ds = ActivationDataset(h5_path, split='test', normalize=True, precompute_stats=False)

    # Share normalization stats from training set
    val_ds.set_normalization_stats(train_ds.layer_means, train_ds.layer_stds)
    test_ds.set_normalization_stats(train_ds.layer_means, train_ds.layer_stds)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_ds,
        'val_dataset': val_ds,
        'test_dataset': test_ds,
    }
