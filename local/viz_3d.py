"""Plotly-based 3D robot graph visualization.

Shows the robot as a graph: spheres for joints, lines for links.
Returns plotly.Figure suitable for Gradio output.
"""
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _edge_segments(positions, edges):
    """Edges as broken-line lists for plotly Scatter3d (lines mode)."""
    pos = _to_np(positions)
    e = _to_np(edges)
    xs, ys, zs = [], [], []
    for k in range(e.shape[1]):
        i, j = int(e[0, k]), int(e[1, k])
        if i >= j:  # plot each undirected edge once
            continue
        xs += [pos[i, 0], pos[j, 0], None]
        ys += [pos[i, 1], pos[j, 1], None]
        zs += [pos[i, 2], pos[j, 2], None]
    return xs, ys, zs


def plot_one(positions, edges, name='robot', color='royalblue'):
    pos = _to_np(positions)
    ex, ey, ez = _edge_segments(positions, edges)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode='lines',
        line=dict(color=color, width=8), showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='markers+text',
        marker=dict(size=7, color=color),
        text=[f'J{i}' for i in range(len(pos))],
        textposition='top center',
        name=name,
    ))
    _set_scene(fig)
    fig.update_layout(title=name, height=520, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def plot_two(pos1, edges1, name1, pos2, edges2, name2,
             color1='royalblue', color2='crimson'):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=[name1, name2],
        horizontal_spacing=0.02,
    )
    for col, (pos, edges, color) in enumerate(
        [(pos1, edges1, color1), (pos2, edges2, color2)], 1
    ):
        p = _to_np(pos)
        ex, ey, ez = _edge_segments(pos, edges)
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode='lines',
            line=dict(color=color, width=8), showlegend=False,
        ), row=1, col=col)
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2],
            mode='markers+text',
            marker=dict(size=7, color=color),
            text=[f'J{i}' for i in range(len(p))],
            textposition='top center',
            showlegend=False,
        ), row=1, col=col)
    _set_scene(fig, n_scenes=2)
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _set_scene(fig, n_scenes=1, lim=1.0):
    scene_dict = dict(
        xaxis=dict(range=[-lim, lim], title='x'),
        yaxis=dict(range=[-lim, lim], title='y'),
        zaxis=dict(range=[0, 1.5 * lim], title='z'),
        aspectmode='cube',
        camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
    )
    if n_scenes == 1:
        fig.update_layout(scene=scene_dict)
    else:
        fig.update_layout(scene=scene_dict, scene2=scene_dict)
