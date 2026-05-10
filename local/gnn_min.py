"""Minimal GNN encoder/decoder that handles arbitrary node counts.

Encoder: per-node MLP -> mean pool -> z (invariant to node count, NOT yet to SE(3))
Decoder: z + per-robot init positions -> message passing -> per-joint angle

This is a placeholder to validate the cross-DOF graph transfer pipeline.
SE(3) equivariance is added later (see notebooks/04).
"""
import torch
import torch.nn as nn


class MeanGNNEncoder(nn.Module):
    """Pose graph -> latent z. Uses node positions + edges via message passing."""

    def __init__(self, z_dim: int = 16, h_dim: int = 64):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(3, h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )
        self.msg = nn.Sequential(
            nn.Linear(2 * h_dim + 1, h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )
        self.readout = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, z_dim),
        )

    def forward(self, positions: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # positions: [N, 3], edge_index: [2, E]
        h = self.in_proj(positions)                    # [N, h]
        i, j = edge_index
        d = (positions[i] - positions[j]).norm(dim=-1, keepdim=True)  # [E,1]
        m = self.msg(torch.cat([h[i], h[j], d], dim=-1))  # [E, h]
        N = h.size(0)
        m_agg = torch.zeros(N, m.size(-1), device=h.device).index_add_(0, i, m)
        h = h + self.update(torch.cat([h, m_agg], dim=-1))
        z = self.readout(h.mean(0))                    # [z_dim]
        return z


class ChainDecoder(nn.Module):
    """z -> per-joint angle for a chain robot of any DOF.
    Uses positional encoding so the same z gives consistent semantics across robots."""

    def __init__(self, z_dim: int = 16, h_dim: int = 64, max_joints: int = 16):
        super().__init__()
        self.pe = nn.Embedding(max_joints, h_dim)
        self.proj = nn.Linear(z_dim, h_dim)
        self.body = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim), nn.SiLU(),
            nn.Linear(h_dim, h_dim), nn.SiLU(),
        )
        self.head = nn.Linear(h_dim, 1)

    def forward(self, z: torch.Tensor, n_joints: int) -> torch.Tensor:
        idx = torch.arange(n_joints, device=z.device)
        # normalize position to [0, 1] so angle semantics scale across DOF
        pe = self.pe(idx) * (1.0 / max(1, n_joints))
        z_node = self.proj(z).unsqueeze(0).expand(n_joints, -1)
        h = self.body(torch.cat([z_node, pe], dim=-1))
        ang = torch.tanh(self.head(h)).squeeze(-1) * 1.5  # [-1.5, 1.5] rad
        return ang
