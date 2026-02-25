"""
The Observer Model

This is the core of the Observer Hypothesis experiment. The observer is a
transformer-based sequence model that watches the executor's state stream
and learns to predict what will happen next.

Everything the observer learns beyond prediction — self-models, surprise,
preferences, temporal integration — is EMERGENT. We don't train for it.
We only look for it after training.

Architecture:
    State packets → Embedding → Transformer (causal) → Prediction heads

The observer is the "mind" watching the "body."
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class StatePacketEmbedder(nn.Module):
    """Embeds a state packet into a single vector.

    Takes all the components of a state packet (env obs, hidden states,
    output logits, action, reward) and combines them into a unified
    representation.

    This is analogous to how consciousness unifies disparate sensory
    streams into a single moment of experience.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, act_dim: int,
                 n_hidden_layers: int, embed_dim: int = 128):
        super().__init__()

        self.act_dim = act_dim

        # Embed each component
        self.obs_embed = nn.Linear(obs_dim, embed_dim // 4)
        self.hidden_embed = nn.Linear(hidden_dim * n_hidden_layers, embed_dim // 4)
        self.logits_embed = nn.Linear(act_dim, embed_dim // 8)
        self.action_embed = nn.Linear(act_dim, embed_dim // 8)  # expects one-hot
        self.reward_embed = nn.Linear(1, embed_dim // 8)
        self.done_embed = nn.Linear(1, embed_dim // 8)

        # Fuse into single embedding
        total_parts = (embed_dim // 4) * 2 + (embed_dim // 8) * 4
        self.fuse = nn.Sequential(
            nn.Linear(total_parts, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.embed_dim = embed_dim

    def forward(self, env_obs, hidden_states, output_logits, actions,
                rewards, dones):
        """Embed a batch of state packets.

        All inputs have shape (batch, seq_len, feature_dim).
        Output: (batch, seq_len, embed_dim)
        """
        # Concatenate hidden states from all layers
        hidden_cat = torch.cat(hidden_states, dim=-1)

        # Convert action index to one-hot if needed
        if actions.shape[-1] != self.act_dim:
            # actions is (batch, seq_len, 1) integer index → one-hot
            act_idx = actions.long().squeeze(-1)
            actions_onehot = torch.zeros(
                *act_idx.shape, self.act_dim, device=actions.device
            )
            actions_onehot.scatter_(-1, act_idx.unsqueeze(-1), 1.0)
        else:
            actions_onehot = actions

        # Embed each stream
        obs_e = self.obs_embed(env_obs)
        hid_e = self.hidden_embed(hidden_cat)
        log_e = self.logits_embed(output_logits)
        act_e = self.action_embed(actions_onehot)
        rew_e = self.reward_embed(rewards)
        don_e = self.done_embed(dones)

        # Concatenate and fuse
        combined = torch.cat([obs_e, hid_e, log_e, act_e, rew_e, don_e], dim=-1)
        return self.fuse(combined)


class ObserverTransformer(nn.Module):
    """The Observer: a transformer that watches the executor's state stream.

    Uses causal (autoregressive) attention — the observer at time t can only
    see packets from times 0..t, not the future. This enforces temporal
    causality: the observer experiences time flowing forward, just as
    consciousness does.

    The observer is trained ONLY on prediction: given packets 0..t, predict
    what happens at t+1. All other properties (self-model, surprise,
    preferences) are emergent.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        act_dim: int,
        n_hidden_layers: int = 2,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Embed state packets
        self.embedder = StatePacketEmbedder(
            obs_dim, hidden_dim, act_dim, n_hidden_layers, embed_dim
        )

        # Positional encoding (learnable)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer encoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
        )

        # Prediction heads (trained objectives)
        self.action_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, act_dim),
        )

        self.obs_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, obs_dim),
        )

        # State summary head (compression — useful for probing)
        self.state_summarizer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
        )

        # Store config for saving/loading
        self.config = {
            'obs_dim': obs_dim,
            'hidden_dim': hidden_dim,
            'act_dim': act_dim,
            'n_hidden_layers': n_hidden_layers,
            'embed_dim': embed_dim,
            'n_heads': n_heads,
            'n_transformer_layers': n_transformer_layers,
            'max_seq_len': max_seq_len,
            'dropout': dropout,
        }

    def _generate_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """Generate causal attention mask.

        The observer at time t can only attend to times 0..t.
        This enforces the arrow of time in the observer's experience.
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        return mask

    def forward(self, env_obs, hidden_states, output_logits, actions,
                rewards, dones, return_hidden=False):
        """Forward pass: observe state packets, predict next states.

        Args:
            env_obs: (batch, seq_len, obs_dim)
            hidden_states: list of (batch, seq_len, hidden_dim)
            output_logits: (batch, seq_len, act_dim)
            actions: (batch, seq_len, act_dim)
            rewards: (batch, seq_len, 1)
            dones: (batch, seq_len, 1)
            return_hidden: if True, also return transformer hidden states

        Returns:
            predicted_actions: (batch, seq_len, act_dim) — prediction for t+1
            predicted_obs: (batch, seq_len, obs_dim) — prediction for t+1
            state_summary: (batch, seq_len, embed_dim//4) — compressed state
            [hidden_states]: optional transformer hidden states for probing
        """
        batch_size, seq_len = env_obs.shape[:2]
        device = env_obs.device

        # Embed state packets
        x = self.embedder(env_obs, hidden_states, output_logits,
                          actions, rewards, dones)

        # Add positional encoding
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # Apply causal transformer
        causal_mask = self._generate_causal_mask(seq_len, device)
        hidden = self.transformer(x, mask=causal_mask)

        # Prediction heads
        predicted_actions = self.action_predictor(hidden)
        predicted_obs = self.obs_predictor(hidden)
        state_summary = self.state_summarizer(hidden)

        if return_hidden:
            return predicted_actions, predicted_obs, state_summary, hidden

        return predicted_actions, predicted_obs, state_summary

    def get_prediction_at(self, env_obs, hidden_states, output_logits,
                          actions, rewards, dones, timestep: int = -1):
        """Get the observer's prediction at a specific timestep.

        Useful for probing: what does the observer "think" will happen next
        at a particular moment?
        """
        pred_actions, pred_obs, summary, hidden = self.forward(
            env_obs, hidden_states, output_logits, actions, rewards, dones,
            return_hidden=True
        )

        return {
            'predicted_action': pred_actions[:, timestep],
            'predicted_obs': pred_obs[:, timestep],
            'state_summary': summary[:, timestep],
            'hidden_state': hidden[:, timestep],
        }
