"""
Experiment 5 Configuration

All hyperparameters, paths, and model specifications in one place.
"""

from pathlib import Path
import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get('EXP5_DATA_DIR', str(BASE_DIR / "data")))
ACTIVATIONS_PATH = DATA_DIR / "activations.h5"
ACTIVATIONS_MEDIUM_PATH = DATA_DIR / "activations_medium.h5"
RESULTS_DIR = DATA_DIR / "results"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

# ── Executor (GPT-2) ──────────────────────────────────────────────────

EXECUTOR_MODEL = "gpt2"                # GPT-2 Small (124M)
EXECUTOR_MODEL_MEDIUM = "gpt2-medium"  # GPT-2 Medium (355M) for Probe 1
EXECUTOR_N_LAYERS = 12                 # 12 transformer layers
EXECUTOR_N_HEADS = 12                  # 12 attention heads per layer
EXECUTOR_D_MODEL = 768                 # Hidden dimension
EXECUTOR_N_CHECKPOINTS = 13            # 13 residual stream checkpoints (pre + post each layer)
EXECUTOR_MAX_CTX = 1024                # GPT-2 max context length

# ── Data ───────────────────────────────────────────────────────────────

SEQ_LEN = 128                          # Tokens per sequence
N_SEQUENCES = 2000                     # Total sequences to extract
N_PILE_SEQUENCES = 1500                # From The Pile (diverse)
N_GARDEN_PATH = 200                    # Garden-path sentences
N_DOMAIN_SWITCH = 150                  # Domain-switch sequences
N_REASONING = 150                      # Reasoning prompts

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ── Observer Architecture ─────────────────────────────────────────────

# Per-layer projection: 768 -> 42 per layer, 13 layers * 42 = 546 -> pad to 512
OBSERVER_PROJ_DIM = 42                 # Per-layer projection output
OBSERVER_D_MODEL = 512                 # Transformer hidden dimension
OBSERVER_N_LAYERS = 6                  # Transformer layers
OBSERVER_N_HEADS = 8                   # Attention heads
OBSERVER_MAX_SEQ_LEN = 128            # Max sequence length
OBSERVER_DROPOUT = 0.1
OBSERVER_FF_DIM = 2048                 # Feedforward dimension (4x d_model)

# Prediction target
PREDICT_LAYER = -1                     # -1 = final layer residual stream

# ── Training ───────────────────────────────────────────────────────────

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
N_EPOCHS = 20
GRAD_CLIP = 1.0
WARMUP_STEPS = 200
MIN_LR_RATIO = 0.1                    # Minimum LR as fraction of peak

# ── Probes ─────────────────────────────────────────────────────────────

N_SEED_OBSERVERS = 5                   # For Probe 5 (emergent preferences RSA)
TEMPORAL_WINDOWS = [1, 4, 16, 64, 128] # For Probe 3 (temporal integration)
SELF_MODEL_P_THRESHOLD = 0.05          # For Probe 1 (significance)

# ── Device ─────────────────────────────────────────────────────────────

def get_device():
    """Get the best available device. Prefer CUDA over MPS."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
