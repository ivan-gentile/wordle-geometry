"""Contrastive / metric-learned geometries from Relation-B similarity S (M2).

Two objectives, both producing a frozen E: word -> R^d aligned to the word list:

  train_reconstruction(S, d)
      Smooth (nonlinear-capable) MDS: learn E so an RBF kernel similarity of E
      matches S directly (MSE). Faithful to the GLOBAL cell structure that the
      cell-centroid controller reads.

  train_supcon(S, d)
      SupCon-style metric learning on thresholded pairs: per anchor, the top-k
      highest-S words are positives, sampled low-S words are negatives; pull
      positives together and push negatives apart in cosine space. Adapted from
      the CLXAI SupConLoss. Shapes tight LOCAL neighborhoods.

Determinism: seeds fix torch + numpy RNG; training is full-batch (no shuffle
nondeterminism) so repeated runs match.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def reconstruction_loss_value(E: np.ndarray, S: np.ndarray,
                              tau: Optional[float] = None) -> float:
    """MSE between the RBF kernel similarity of E and the target S (numpy).

    The length-scale ``tau`` defaults to the embedding dimension, which keeps
    ``exp(-||E_a-E_b||^2 / tau)`` in a useful (0,1) range for random init
    (a unit length-scale saturates to 0 for d>>1 and kills the gradient).
    """
    if tau is None:
        tau = float(E.shape[1])
    d2 = np.sum((E[:, None, :] - E[None, :, :]) ** 2, axis=-1)
    sim = np.exp(-d2 / tau)
    return float(np.mean((sim - S) ** 2))


def train_reconstruction(
    S: np.ndarray, d: int = 64, epochs: int = 1500, lr: float = 0.05,
    tau: Optional[float] = None, seed: int = 0, verbose: bool = False,
) -> np.ndarray:
    """Learn E so exp(-||E_a - E_b||^2 / tau) ~= S[a,b] (full-batch Adam).

    ``tau`` (length-scale) defaults to ``d`` so the kernel is well-scaled at
    random init; without it the kernel saturates to 0 and training stalls.
    """
    _set_seed(seed)
    n = S.shape[0]
    if tau is None:
        tau = float(d)
    St = torch.tensor(S, dtype=torch.float32)
    E = torch.randn(n, d, requires_grad=True)
    opt = torch.optim.Adam([E], lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        d2 = torch.cdist(E, E) ** 2
        sim = torch.exp(-d2 / tau)
        loss = ((sim - St) ** 2).mean()
        loss.backward()
        opt.step()
        if verbose and ep % 200 == 0:
            print(f"  recon ep {ep} loss {loss.item():.5f}")
    return E.detach().numpy().astype(np.float64)


class _SupConOnPairs(nn.Module):
    """SupCon-style loss using a precomputed positive mask (adapted from CLXAI).

    For each anchor i, positives are the entries with pos_mask[i, j] = 1; all
    other (non-self) entries are treated as negatives via the denominator.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        feats = nn.functional.normalize(features, dim=1)
        sim = feats @ feats.T / self.temperature
        n = feats.shape[0]
        self_mask = torch.eye(n, device=feats.device)
        logits = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        pos = pos_mask * (1 - self_mask)
        num_pos = torch.clamp(pos.sum(dim=1), min=1.0)
        mean_log_prob_pos = (pos * log_prob).sum(dim=1) / num_pos
        # only count anchors that have at least one positive
        has_pos = (pos.sum(dim=1) > 0).float()
        return -(mean_log_prob_pos * has_pos).sum() / torch.clamp(has_pos.sum(), min=1.0)


def train_supcon(
    S: np.ndarray, d: int = 64, epochs: int = 1500, lr: float = 0.05,
    temperature: float = 0.1, k_pos: int = 8, n_neg: int = 0,
    seed: int = 0, verbose: bool = False,
) -> np.ndarray:
    """SupCon metric learning on thresholded high-S/low-S pairs.

    ``k_pos`` highest-S words per anchor are positives. ``n_neg`` is accepted for
    API symmetry; negatives are handled implicitly by the SupCon denominator
    (all non-positive, non-self entries). Full-batch.
    """
    _set_seed(seed)
    n = S.shape[0]
    # Build positive mask: top-k_pos by similarity per row (excluding self).
    Sm = S.copy()
    np.fill_diagonal(Sm, -np.inf)
    pos_idx = np.argsort(Sm, axis=1)[:, -k_pos:]      # (n, k_pos)
    pos_mask = np.zeros((n, n), dtype=np.float32)
    rows = np.repeat(np.arange(n), k_pos)
    pos_mask[rows, pos_idx.reshape(-1)] = 1.0
    pos_mask = np.maximum(pos_mask, pos_mask.T)        # symmetric positives

    pos_t = torch.tensor(pos_mask)
    E = torch.randn(n, d, requires_grad=True)
    loss_fn = _SupConOnPairs(temperature=temperature)
    opt = torch.optim.Adam([E], lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn(E, pos_t)
        loss.backward()
        opt.step()
        if verbose and ep % 200 == 0:
            print(f"  supcon ep {ep} loss {loss.item():.5f}")
    return E.detach().numpy().astype(np.float64)
