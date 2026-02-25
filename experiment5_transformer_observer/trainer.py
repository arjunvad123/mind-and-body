"""
Observer Trainer

Trains the observer transformer on GPT-2 activation prediction.
The observer's ONLY objective is prediction: given the residual stream
at positions 0..T, predict the final-layer residual at position T+1.

Also trains the linear baseline and shuffled control for comparison.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

from . import config
from .observer_model import ObserverTransformer, LinearBaseline
from .dataset import ActivationDataset, ShuffledActivationDataset, create_dataloaders


class ObserverTrainer:
    """Trains the observer on next-position residual stream prediction.

    Loss = MSE(observer_predict(resid[0:T]), resid[T+1, final_layer])

    The observer develops its internal representations entirely in service
    of this prediction objective. What those representations mean — whether
    they constitute a self-model, whether they correlate with computational
    complexity, whether they integrate information — that's for the probes
    to determine AFTER training.
    """

    def __init__(
        self,
        model: nn.Module = None,
        lr: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        device: torch.device = None,
    ):
        self.device = device or config.DEVICE
        self.model = model or ObserverTransformer()
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        total, trainable = self.model.count_parameters()
        print(f"Observer parameters: {total:,} total, {trainable:,} trainable")

    def _create_scheduler(self, n_epochs: int, steps_per_epoch: int):
        """Cosine schedule with linear warmup."""
        total_steps = n_epochs * steps_per_epoch
        warmup_steps = min(config.WARMUP_STEPS, total_steps // 5)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return config.MIN_LR_RATIO + (1 - config.MIN_LR_RATIO) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(
        self,
        train_loader,
        val_loader=None,
        n_epochs: int = config.N_EPOCHS,
        save_path: str = None,
        print_every: int = 2,
    ):
        """Train the observer.

        Args:
            train_loader: DataLoader of (input_resid, target_resid) pairs
            val_loader: Optional validation DataLoader
            n_epochs: Number of training epochs
            save_path: Where to save the best model
            print_every: Logging frequency

        Returns:
            history dict with train/val losses
        """
        print(f"Training observer for {n_epochs} epochs")
        print(f"Device: {self.device}")
        print("-" * 60)

        scheduler = self._create_scheduler(n_epochs, len(train_loader))
        loss_fn = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        for epoch in range(n_epochs):
            # ── Train ───────────────────────────────────────────
            self.model.train()
            epoch_losses = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}",
                        leave=False) if len(train_loader) > 5 else train_loader

            for input_resid, target_resid in pbar:
                # input_resid: (batch, n_checkpoints, seq_len-1, d_model)
                # target_resid: (batch, seq_len-1, d_model)
                input_resid = input_resid.to(self.device)
                target_resid = target_resid.to(self.device)

                # Forward pass
                predictions = self.model(input_resid)  # (batch, seq_len-1, d_model)

                # Loss: MSE between predicted and actual next-position residual
                loss = loss_fn(predictions, target_resid)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP)
                self.optimizer.step()
                scheduler.step()

                epoch_losses.append(loss.item())

                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix(loss=f"{loss.item():.6f}")

            avg_train_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_train_loss)

            # ── Validate ────────────────────────────────────────
            avg_val_loss = None
            if val_loader is not None:
                avg_val_loss = self._validate(val_loader, loss_fn)
                history['val_loss'].append(avg_val_loss)

                # Save best model
                if save_path and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self._save(save_path)

            # ── Log ─────────────────────────────────────────────
            if (epoch + 1) % print_every == 0:
                msg = f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.6f}"
                if avg_val_loss is not None:
                    msg += f" | Val Loss: {avg_val_loss:.6f}"
                msg += f" | LR: {scheduler.get_last_lr()[0]:.2e}"
                print(msg)

        # Save final model if no validation
        if save_path and val_loader is None:
            self._save(save_path)

        return history

    def _validate(self, val_loader, loss_fn):
        """Run validation pass."""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for input_resid, target_resid in val_loader:
                input_resid = input_resid.to(self.device)
                target_resid = target_resid.to(self.device)

                predictions = self.model(input_resid)
                loss = loss_fn(predictions, target_resid)
                val_losses.append(loss.item())

        return np.mean(val_losses)

    def _save(self, save_path):
        """Save model checkpoint."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)


def train_observer(
    h5_path: str = None,
    save_path: str = None,
    device: torch.device = None,
) -> dict:
    """Train the primary observer model.

    Returns:
        Training history dict
    """
    h5_path = h5_path or str(config.ACTIVATIONS_PATH)
    save_path = save_path or str(config.CHECKPOINTS_DIR / 'observer.pt')
    device = device or config.DEVICE

    dataloaders = create_dataloaders(h5_path)

    model = ObserverTransformer()
    trainer = ObserverTrainer(model=model, device=device)

    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        save_path=save_path,
    )

    return {
        'history': history,
        'model': trainer.model,
        'train_dataset': dataloaders['train_dataset'],
        'val_dataset': dataloaders['val_dataset'],
        'test_dataset': dataloaders['test_dataset'],
    }


def train_linear_baseline(
    h5_path: str = None,
    save_path: str = None,
    device: torch.device = None,
) -> dict:
    """Train the linear baseline model.

    Single matrix W predicting S(T+1) = W @ S(T).
    If this matches the observer, temporal modeling isn't needed.
    """
    h5_path = h5_path or str(config.ACTIVATIONS_PATH)
    save_path = save_path or str(config.CHECKPOINTS_DIR / 'linear_baseline.pt')
    device = device or config.DEVICE

    print("\n" + "=" * 60)
    print("Training Linear Baseline")
    print("=" * 60)

    dataloaders = create_dataloaders(h5_path)

    model = LinearBaseline()
    trainer = ObserverTrainer(model=model, device=device)

    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        save_path=save_path,
    )

    return {'history': history, 'model': trainer.model}


def train_shuffled_observer(
    h5_path: str = None,
    save_path: str = None,
    device: torch.device = None,
) -> dict:
    """Train observer on temporally shuffled data.

    Controls for learning activation statistics without temporal dynamics.
    If this matches the real observer, temporal dynamics don't matter.
    """
    from torch.utils.data import DataLoader

    h5_path = h5_path or str(config.ACTIVATIONS_PATH)
    save_path = save_path or str(config.CHECKPOINTS_DIR / 'shuffled_observer.pt')
    device = device or config.DEVICE

    print("\n" + "=" * 60)
    print("Training Shuffled Observer (Control)")
    print("=" * 60)

    # Create base dataset and wrap in shuffled version
    base_train = ActivationDataset(h5_path, split='train')
    base_val = ActivationDataset(h5_path, split='val', precompute_stats=False)
    base_val.set_normalization_stats(base_train.layer_means, base_train.layer_stds)

    shuffled_train = ShuffledActivationDataset(base_train)
    shuffled_val = ShuffledActivationDataset(base_val)

    train_loader = DataLoader(shuffled_train, batch_size=config.BATCH_SIZE,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(shuffled_val, batch_size=config.BATCH_SIZE,
                            shuffle=False, pin_memory=True)

    model = ObserverTransformer()
    trainer = ObserverTrainer(model=model, device=device)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_path=save_path,
    )

    return {'history': history, 'model': trainer.model}


def train_seed_observers(
    h5_path: str = None,
    n_seeds: int = config.N_SEED_OBSERVERS,
    device: torch.device = None,
) -> list:
    """Train multiple observers with different random seeds.

    For Probe 5 (emergent preferences): RSA between observers
    trained on the same data with different initializations.
    """
    h5_path = h5_path or str(config.ACTIVATIONS_PATH)
    device = device or config.DEVICE

    print("\n" + "=" * 60)
    print(f"Training {n_seeds} Seed Observers (for RSA)")
    print("=" * 60)

    dataloaders = create_dataloaders(h5_path)
    results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        save_path = str(config.CHECKPOINTS_DIR / f'observer_seed{seed}.pt')

        model = ObserverTransformer()
        trainer = ObserverTrainer(model=model, device=device)

        history = trainer.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            save_path=save_path,
            print_every=5,
        )

        results.append({
            'seed': seed,
            'history': history,
            'model': trainer.model,
            'save_path': save_path,
        })

    return results


def load_model(checkpoint_path: str, device: torch.device = None) -> nn.Module:
    """Load a saved model from checkpoint."""
    device = device or config.DEVICE
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint['config']

    if model_config.get('type') == 'linear_baseline':
        model = LinearBaseline(d_model=model_config['d_model'])
    else:
        model = ObserverTransformer(**{
            k: v for k, v in model_config.items() if k != 'type'
        })

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model
