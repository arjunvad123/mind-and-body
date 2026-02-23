"""
Dual-Path Observer

A modified observer with two explicit processing paths:

FAST PATH (System 1 / First Thought):
  - Single layer, no self-attention
  - Direct mapping from encoded state to prediction
  - Analogous to gut feeling, pattern matching, amygdala fast route
  - Produces the "first thought"

SLOW PATH (System 2 / Deliberation):
  - Multiple layers with self-attention
  - Can integrate information across the full context window
  - Analogous to prefrontal cortex, working memory, conscious reasoning
  - Produces the "deliberated thought"

Both paths receive the same input. Both are trained on the same objective.
The question: do they develop different competency profiles?

Our hypothesis (from Arjun's insight): the fast path will be MORE accurate
than the slow path, mirroring how human System 1 often outperforms System 2.
"""

import torch
import torch.nn as nn
import math

from experiment1_separation.observer.model import StatePacketEmbedder


class DualPathObserver(nn.Module):
    """Observer with separate fast and slow processing paths.

    The fast path is the "first thought" — one layer, instant.
    The slow path is "deliberation" — multiple layers, integrative.

    We compare their predictions to test whether consciousness
    (slow deliberation) actually improves on unconscious pattern
    matching (fast response).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        act_dim: int,
        n_hidden_layers: int = 2,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_slow_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.act_dim = act_dim
        self.obs_dim = obs_dim

        # Shared embedder (both paths see the same input)
        self.embedder = StatePacketEmbedder(
            obs_dim, hidden_dim, act_dim, n_hidden_layers, embed_dim
        )

        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # ---- FAST PATH (System 1) ----
        # Single feedforward layer: see input → predict output. No deliberation.
        self.fast_path = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, act_dim),
        )

        # ---- SLOW PATH (System 2) ----
        # Full transformer: see full context → deliberate → predict
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.slow_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_slow_layers,
        )
        self.slow_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, act_dim),
        )

        # ---- ITERATIVE DELIBERATION ----
        # For K-pass deliberation: observer re-processes its own output
        self.deliberation_integrator = nn.Sequential(
            nn.Linear(embed_dim + act_dim, embed_dim),
            nn.GELU(),
        )

        # Store config
        self.config = {
            'obs_dim': obs_dim, 'hidden_dim': hidden_dim, 'act_dim': act_dim,
            'n_hidden_layers': n_hidden_layers, 'embed_dim': embed_dim,
            'n_heads': n_heads, 'n_slow_layers': n_slow_layers,
            'max_seq_len': max_seq_len, 'dropout': dropout,
        }

    def _causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )

    def forward(self, env_obs, hidden_states, output_logits, actions,
                rewards, dones, n_deliberation_passes=0):
        """Forward pass through both paths.

        Args:
            ... (same as ObserverTransformer)
            n_deliberation_passes: number of additional slow-path iterations.
                0 = single slow pass. K > 0 = observer re-processes K times.

        Returns:
            fast_prediction: (batch, seq_len, act_dim) — first thought
            slow_prediction: (batch, seq_len, act_dim) — deliberated thought
            fast_hidden: (batch, seq_len, embed_dim) — fast path representations
            slow_hidden: (batch, seq_len, embed_dim) — slow path representations
        """
        batch_size, seq_len = env_obs.shape[:2]
        device = env_obs.device

        # Shared embedding
        x = self.embedder(env_obs, hidden_states, output_logits,
                          actions, rewards, dones)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # ---- FAST PATH ----
        # Direct prediction: no context integration, just pattern match
        fast_prediction = self.fast_path(x)
        fast_hidden = x  # the embedding IS the fast path representation

        # ---- SLOW PATH ----
        causal_mask = self._causal_mask(seq_len, device)
        slow_hidden = self.slow_transformer(x, mask=causal_mask)
        slow_prediction = self.slow_predictor(slow_hidden)

        # ---- ITERATIVE DELIBERATION ----
        # The observer re-processes its own prediction K times
        # This simulates "thinking about it more" — System 2 working harder
        current_prediction = slow_prediction
        for k in range(n_deliberation_passes):
            # Integrate previous prediction back into representation
            augmented = torch.cat([slow_hidden, current_prediction], dim=-1)
            refined = self.deliberation_integrator(augmented)
            # Re-process through slow path
            slow_hidden = self.slow_transformer(refined, mask=causal_mask)
            current_prediction = self.slow_predictor(slow_hidden)

        if n_deliberation_passes > 0:
            slow_prediction = current_prediction

        return fast_prediction, slow_prediction, fast_hidden, slow_hidden

    def get_fast_prediction(self, env_obs, hidden_states, output_logits,
                            actions, rewards, dones):
        """Get ONLY the first-thought prediction. No deliberation."""
        x = self.embedder(env_obs, hidden_states, output_logits,
                          actions, rewards, dones)
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        return self.fast_path(x)

    def get_deliberated_prediction(self, env_obs, hidden_states, output_logits,
                                    actions, rewards, dones, n_passes=0):
        """Get the slow-path prediction with optional extra deliberation."""
        _, slow_pred, _, _ = self.forward(
            env_obs, hidden_states, output_logits, actions, rewards, dones,
            n_deliberation_passes=n_passes
        )
        return slow_pred
