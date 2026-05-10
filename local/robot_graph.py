"""Robot kinematic structures as graphs.

Each robot is defined as:
- nodes: joints (with axis, limits, type)
- edges: parent-child links
- forward kinematics: angles -> 3D positions

This is the most minimal representation: serial revolute chains.
URDF loading can be added later (yourdfpy / pytorch-kinematics).
"""
from dataclasses import dataclass
import torch


@dataclass
class RobotGraph:
    name: str
    n_joints: int
    edges: torch.Tensor        # [2, E] (i, j) parent-child both directions
    link_lengths: torch.Tensor # [N]
    joint_axes: torch.Tensor   # [N, 3]
    base_pos: torch.Tensor     # [3]
    joint_limits: torch.Tensor # [N, 2]


def serial_chain(n: int, link_len: float = 0.15, name: str = None) -> RobotGraph:
    """Revolute serial chain. Joint i rotates around axis i%3 (alternating)."""
    edges = []
    for i in range(n - 1):
        edges += [(i, i + 1), (i + 1, i)]
    edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros(2, 0, dtype=torch.long)

    axes = torch.zeros(n, 3)
    for i in range(n):
        axes[i, i % 3] = 1.0

    return RobotGraph(
        name=name or f'{n}-DOF chain',
        n_joints=n,
        edges=edge_index,
        link_lengths=torch.full((n,), link_len),
        joint_axes=axes,
        base_pos=torch.zeros(3),
        joint_limits=torch.tensor([[-3.14, 3.14]] * n),
    )


# Preset registry
ROBOTS = {
    'Franka-like 7-DOF': lambda: serial_chain(7, 0.12, 'Franka-like 7-DOF'),
    'UR5-like 6-DOF':    lambda: serial_chain(6, 0.15, 'UR5-like 6-DOF'),
    'Custom 5-DOF':      lambda: serial_chain(5, 0.18, 'Custom 5-DOF'),
    'Custom 8-DOF':      lambda: serial_chain(8, 0.10, 'Custom 8-DOF'),
    'Mini 3-DOF':        lambda: serial_chain(3, 0.25, 'Mini 3-DOF'),
}


def rodrigues(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Differentiable axis-angle rotation matrix [3,3]."""
    a = axis / (axis.norm() + 1e-9)
    c = torch.cos(angle); s = torch.sin(angle); t = 1 - c
    x, y, z = a[0], a[1], a[2]
    R = torch.stack([
        torch.stack([t*x*x + c,    t*x*y - s*z,  t*x*z + s*y]),
        torch.stack([t*x*y + s*z,  t*y*y + c,    t*y*z - s*x]),
        torch.stack([t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]),
    ])
    return R


def fk_chain(robot: RobotGraph, angles: torch.Tensor) -> torch.Tensor:
    """Forward kinematics. angles [N] -> positions [N+1, 3] (base + N joint endpoints)."""
    assert angles.shape[0] == robot.n_joints, f'expected {robot.n_joints} angles, got {angles.shape[0]}'
    R = torch.eye(3)
    p = robot.base_pos.clone()
    out = [p]
    for i in range(robot.n_joints):
        Ri = rodrigues(robot.joint_axes[i], angles[i])
        R = R @ Ri
        p = p + R @ torch.tensor([0., 0., float(robot.link_lengths[i])])
        out.append(p)
    return torch.stack(out)


def graph_to_pyg_edges(robot: RobotGraph) -> torch.Tensor:
    """Returns edge_index including base node (index 0) + N joint endpoints (1..N).
    Base is connected to joint 1 in the position graph."""
    base_edges = []
    for i in range(robot.n_joints):
        base_edges += [(i, i + 1), (i + 1, i)]
    return torch.tensor(base_edges, dtype=torch.long).t()
