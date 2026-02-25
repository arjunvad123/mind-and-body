"""
Experiment 6 Configuration

All hyperparameters, paths, and model specifications in one place.
"""

from pathlib import Path
import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get('EXP6_DATA_DIR', str(BASE_DIR / "data")))
SYSTEM_DATA_PATH = DATA_DIR / "system_data.h5"
TRAJECTORIES_PATH = DATA_DIR / "executor_trajectories.h5"
TRAJECTORIES_SECONDARY_PATH = DATA_DIR / "executor_trajectories_secondary.h5"
RESULTS_DIR = DATA_DIR / "results"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

# ── Dynamical Systems ─────────────────────────────────────────────────

SYSTEM_DIM = 8            # All systems padded/projected to 8D uniform input
N_SYSTEMS = 8             # Number of distinct dynamical system types
N_TRAJECTORIES_PER_SYSTEM = 100   # Trajectories per system type
TRAJECTORY_LENGTH = 500   # Timesteps per trajectory
DT = 0.02                 # Integration timestep
TOTAL_TRAJECTORIES = 800  # N_SYSTEMS * N_TRAJECTORIES_PER_SYSTEM
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SYSTEM_TYPES = [
    'lorenz', 'rossler', 'double_pendulum', 'coupled_oscillators',
    'van_der_pol', 'damped_sine', 'step_function', 'logistic_map'
]

# ── Executor (CfC) ───────────────────────────────────────────────────

EXECUTOR_HIDDEN_SIZE = 64
EXECUTOR_OUTPUT_SIZE = 8       # = SYSTEM_DIM (predict next state)
EXECUTOR_MODE = 'default'      # CfC mode
EXECUTOR_BACKBONE_UNITS = 64
EXECUTOR_BACKBONE_LAYERS = 1

# ── Observer (CfC) ───────────────────────────────────────────────────

OBSERVER_DEFAULT_SIZE = 50     # Default observer hidden neurons
OBSERVER_SIZES = [10, 20, 50, 100, 200]  # For scaling experiment
OBSERVER_OUTPUT_SIZE = 64      # = EXECUTOR_HIDDEN_SIZE (predict next hidden)
OBSERVER_MODE = 'default'
OBSERVER_BACKBONE_UNITS = 64
OBSERVER_BACKBONE_LAYERS = 1

# ── Training ─────────────────────────────────────────────────────────

EXECUTOR_LR = 1e-3
EXECUTOR_EPOCHS = 50
OBSERVER_LR = 1e-3
OBSERVER_EPOCHS = 30
BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_STEPS = 100
MIN_LR_RATIO = 0.1

# ── Probes ───────────────────────────────────────────────────────────

N_SEED_OBSERVERS = 5
TEMPORAL_WINDOWS = [1, 4, 16, 64, 128, 250, 500]
SELF_MODEL_P_THRESHOLD = 0.05
N_DELIBERATION_PASSES = 3
N_PHI_PARTITIONS = 20         # For approximate Phi

# ── Device ───────────────────────────────────────────────────────────

def get_device():
    """Get the best available device. Prefer CUDA over MPS."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
