"""SE(3) invariance/equivariance checker.

Runs random rotation+translation tests on a saved encoder checkpoint
and reports max |z - z_rotated|. Sanity gate before any downstream training.

Usage:
    python local/check_equivariance.py --ckpt checkpoints/gnn_v0.pt --trials 50
"""
import argparse
import sys
import torch
from pathlib import Path

# import notebook 04's model defs by re-defining here (keeps script standalone)
import torch.nn as nn


class EGNNLayer(nn.Module):
    def __init__(self, h_dim, edge_dim=0, hidden=128):
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(2 * h_dim + 1 + edge_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(h_dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, h_dim),
        )

    def forward(self, h, x, edge_index):
        i, j = edge_index
        rel = x[i] - x[j]
        d2 = (rel ** 2).sum(-1, keepdim=True)
        m = self.phi_e(torch.cat([h[i], h[j], d2], dim=-1))
        coord_w = self.phi_x(m)
        x_msg = rel * coord_w
        N = h.size(0)
        x_agg = torch.zeros_like(x).index_add_(0, i, x_msg)
        x_new = x + x_agg / max(1, edge_index.size(1) // N)
        h_agg = torch.zeros(N, m.size(-1), device=h.device).index_add_(0, i, m)
        h_new = h + self.phi_h(torch.cat([h, h_agg], dim=-1))
        return h_new, x_new


class PoseEncoder(nn.Module):
    def __init__(self, n_layers=4, h_dim=64, z_dim=32, n_node_types=21):
        super().__init__()
        self.embed = nn.Embedding(n_node_types, h_dim)
        self.layers = nn.ModuleList([EGNNLayer(h_dim) for _ in range(n_layers)])
        self.readout = nn.Sequential(
            nn.Linear(h_dim, 128), nn.SiLU(),
            nn.Linear(128, z_dim),
        )

    def forward(self, x, node_type, edge_index):
        h = self.embed(node_type)
        for L in self.layers:
            h, x = L(h, x, edge_index)
        return self.readout(h.mean(0))


def random_so3():
    A = torch.randn(3, 3)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diag(R)).unsqueeze(0)


def check(encoder, n_nodes=21, n_trials=50, atol=1e-4):
    encoder.eval()
    diffs = []
    with torch.no_grad():
        for _ in range(n_trials):
            x = torch.randn(n_nodes, 3)
            node_type = torch.arange(n_nodes)
            edge_index = torch.tensor(
                [[i, (i + 1) % n_nodes] for i in range(n_nodes)] +
                [[(i + 1) % n_nodes, i] for i in range(n_nodes)]
            ).t()
            z1 = encoder(x, node_type, edge_index)
            R = random_so3()
            t = torch.randn(3) * 0.5
            x2 = x @ R.t() + t
            z2 = encoder(x2, node_type, edge_index)
            diffs.append((z1 - z2).abs().max().item())
    diffs = torch.tensor(diffs)
    print(f'Trials: {n_trials}')
    print(f'  mean diff: {diffs.mean():.2e}')
    print(f'  max  diff: {diffs.max():.2e}')
    print(f'  tol     : {atol:.2e}')
    passed = (diffs < atol).all().item()
    print('  RESULT  :', 'PASS' if passed else 'FAIL')
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--atol', type=float, default=1e-4)
    ap.add_argument('--n_nodes', type=int, default=21)
    args = ap.parse_args()

    enc = PoseEncoder(n_node_types=args.n_nodes)
    if args.ckpt and Path(args.ckpt).exists():
        sd = torch.load(args.ckpt, map_location='cpu')
        enc.load_state_dict(sd['encoder'] if 'encoder' in sd else sd)
        print(f'Loaded {args.ckpt}')
    else:
        print('No ckpt → testing freshly initialized encoder (still must pass)')

    ok = check(enc, n_nodes=args.n_nodes, n_trials=args.trials, atol=args.atol)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
