"""3D skeleton viewer (matplotlib).
Usage:
    python local/viz_skeleton.py --pose data/sample_pose.npz
    python local/viz_skeleton.py --pose data/sample_pose.npz --frame 30
    python local/viz_skeleton.py --pose data/sample_pose.npz --animate
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.animation import FuncAnimation


def draw(ax, frame, edges, title=None):
    ax.cla()
    valid = ~np.isnan(frame).any(1)
    ax.scatter(frame[valid, 0], frame[valid, 1], frame[valid, 2], s=30)
    for a, b in edges:
        if valid[a] and valid[b]:
            ax.plot(
                [frame[a, 0], frame[b, 0]],
                [frame[a, 1], frame[b, 1]],
                [frame[a, 2], frame[b, 2]],
                'k-', linewidth=1.5,
            )
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    if title:
        ax.set_title(title)
    # Equal aspect (matplotlib 3D doesn't have set_aspect('equal') reliably)
    pts = frame[valid]
    if len(pts) > 0:
        c = pts.mean(0)
        r = max(1e-3, (pts - c).max())
        ax.set_xlim(c[0]-r, c[0]+r)
        ax.set_ylim(c[1]-r, c[1]+r)
        ax.set_zlim(c[2]-r, c[2]+r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose', required=True, help='.npz with keys: keypoints (T,N,3), edges (E,2)')
    ap.add_argument('--frame', type=int, default=None)
    ap.add_argument('--animate', action='store_true')
    args = ap.parse_args()

    d = np.load(args.pose, allow_pickle=True)
    seq = d['keypoints']
    edges = d['edges']
    name = str(d['video_name']) if 'video_name' in d.files else args.pose

    if seq.ndim == 2:
        seq = seq[None]
    print(f'Loaded {name}: seq {seq.shape}, edges {edges.shape}')

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    if args.animate:
        def update(i):
            draw(ax, seq[i], edges, title=f'{name}  frame {i}/{len(seq)}')
        anim = FuncAnimation(fig, update, frames=len(seq), interval=50)
        plt.show()
    else:
        f = args.frame if args.frame is not None else len(seq) // 2
        draw(ax, seq[f], edges, title=f'{name}  frame {f}')
        plt.show()


if __name__ == '__main__':
    main()
