"""
Observer Transformer Model

A small transformer (~15M params) that watches GPT-2's internal activations
token-by-token, layer-by-layer, trained purely on prediction.

Architecture:
    Per-layer projection (768d → 42d per layer, 13 layers → 546d → 512d)
    → Causal Transformer (6 layers, 8 heads, d=512)
    → Prediction Head (512d → 768d, predicting next-position final-layer residual)

The observer must *model* the executor, not memorize it. The capacity bottleneck
(15M vs 124M) forces compression — only the most predictive information survives.
"""

import torch
import torch.nn as nn
import math

from . import config


class PerLayerProjection(nn.Module):
    """Projects each layer's residual stream into a compact representation.

    Each of the 13 residual stream checkpoints (768d) is independently projected
    to a smaller dimension (42d). The results are concatenated (546d) and fused
    to the observer's working dimension (512d).

    This preserves the hierarchical structure: the observer can learn which
    layers are most informative, analogous to consciousness "attending to"
    specific processing levels.
    """

    def __init__(
        self,
        n_checkpoints: int = config.EXECUTOR_N_CHECKPOINTS,  # 13
        d_executor: int = config.EXECUTOR_D_MODEL,            # 768
        proj_dim: int = config.OBSERVER_PROJ_DIM,             # 42
        d_observer: int = config.OBSERVER_D_MODEL,            # 512
    ):
        super().__init__()

        self.n_checkpoints = n_checkpoints
        self.proj_dim = proj_dim

        # Independent projection for each layer
        self.projections = nn.ModuleList([
            nn.Linear(d_executor, proj_dim) for _ in range(n_checkpoints)
        ])

        # Fuse concatenated projections to observer dimension
        concat_dim = n_checkpoints * proj_dim  # 13 * 42 = 546
        self.fuse = nn.Sequential(
            nn.Linear(concat_dim, d_observer),
            nn.LayerNorm(d_observer),
            nn.GELU(),
        )

    def forward(self, residual_stream):
        """
        Args:
            residual_stream: (batch, n_checkpoints, seq_len, d_executor)

        Returns:
            fused: (batch, seq_len, d_observer)
        """
        projections = []
        for i, proj in enumerate(self.projections):
            # (batch, seq_len, d_executor) -> (batch, seq_len, proj_dim)
            layer_proj = proj(residual_stream[:, i, :, :])
            projections.append(layer_proj)

        # (batch, seq_len, n_checkpoints * proj_dim)
        concatenated = torch.cat(projections, dim=-1)

        # (batch, seq_len, d_observer)
        return self.fuse(concatenated)


class ObserverTransformer(nn.Module):
    """The Observer: watches GPT-2's neural activations and predicts what comes next.

    Uses causal (autoregressive) attention — the observer at position t can only
    see activations from positions 0..t. This enforces temporal causality.

    Trained ONLY on prediction: given activations at positions 0..T, predict
    the final-layer residual at position T+1. All other properties (self-model,
    surprise, temporal integration, preferences) are emergent.
    """

    def __init__(
        self,
        n_checkpoints: int = config.EXECUTOR_N_CHECKPOINTS,
        d_executor: int = config.EXECUTOR_D_MODEL,
        proj_dim: int = config.OBSERVER_PROJ_DIM,
        d_model: int = config.OBSERVER_D_MODEL,
        n_layers: int = config.OBSERVER_N_LAYERS,
        n_heads: int = config.OBSERVER_N_HEADS,
        max_seq_len: int = config.OBSERVER_MAX_SEQ_LEN,
        dropout: float = config.OBSERVER_DROPOUT,
        ff_dim: int = config.OBSERVER_FF_DIM,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Per-layer projection (768d per layer → 42d, fused to 512d)
        self.embedder = PerLayerProjection(
            n_checkpoints=n_checkpoints,
            d_executor=d_executor,
            proj_dim=proj_dim,
            d_observer=d_model,
        )

        # Learnable positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Causal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Prediction head: predict next-position final-layer residual
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_executor),
        )

        # Store config for saving/loading
        self.config = {
            'n_checkpoints': n_checkpoints,
            'd_executor': d_executor,
            'proj_dim': proj_dim,
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'max_seq_len': max_seq_len,
            'dropout': dropout,
            'ff_dim': ff_dim,
        }

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform for projections."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _generate_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """Generate causal attention mask.

        The observer at position t can only attend to positions 0..t.
        This enforces the arrow of time.
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1,
        )
        return mask

    def forward(self, residual_stream, return_hidden=False):
        """Forward pass: observe activations, predict next state.

        Args:
            residual_stream: (batch, n_checkpoints, seq_len, d_executor)
                All 13 residual stream checkpoints across the sequence.
            return_hidden: If True, also return transformer hidden states
                for probing.

        Returns:
            predictions: (batch, seq_len, d_executor) — predicted next-position
                final-layer residual
            hidden: (batch, seq_len, d_model) — observer's internal states
                (only if return_hidden=True)
        """
        batch_size = residual_stream.shape[0]
        seq_len = residual_stream.shape[2]
        device = residual_stream.device

        # Project each layer's residual and fuse
        x = self.embedder(residual_stream)  # (batch, seq_len, d_model)

        # Add positional encoding
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # Apply causal transformer
        causal_mask = self._generate_causal_mask(seq_len, device)
        hidden = self.transformer(x, mask=causal_mask)

        # Predict next-position final-layer residual
        predictions = self.prediction_head(hidden)

        if return_hidden:
            return predictions, hidden

        return predictions

    def predict_at(self, residual_stream, position: int = -1):
        """Get the observer's prediction at a specific position.

        Useful for probing: what does the observer "think" will happen next?
        """
        predictions, hidden = self.forward(residual_stream, return_hidden=True)

        return {
            'prediction': predictions[:, position],
            'hidden_state': hidden[:, position],
        }

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class LinearBaseline(nn.Module):
    """Linear baseline: single matrix predicting S(T+1) = W @ S(T).

    If this matches the observer, temporal modeling isn't needed —
    the observer would just be a fancy linear predictor.
    """

    def __init__(
        self,
        d_model: int = config.EXECUTOR_D_MODEL,
    ):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.config = {'d_model': d_model, 'type': 'linear_baseline'}

    def forward(self, residual_stream, return_hidden=False):
        """
        Args:
            residual_stream: (batch, n_checkpoints, seq_len, d_executor)

        Returns:
            predictions: (batch, seq_len, d_executor) — S(T+1) predicted from S(T)
        """
        # Use only the final-layer residual (last checkpoint)
        final_layer = residual_stream[:, -1, :, :]  # (batch, seq_len, d_executor)

        predictions = self.linear(final_layer)

        if return_hidden:
            return predictions, final_layer  # "hidden state" is just the input

        return predictions

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return total, total
