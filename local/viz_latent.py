"""Latent space (z) visualizer — t-SNE / UMAP.

Loads (z, label) pairs from .npz and projects to 2D. Useful for checking
whether human-pose z and robot-pose z form aligned clusters by task.

Usage:
    python local/viz_latent.py --z data/latents.npz --method tsne
    python local/viz_latent.py --z data/latents.npz --method umap --color task
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z', required=True, help='.npz with z (N,D), labels (N,) optional, source (N,) optional')
    ap.add_argument('--method', choices=['tsne', 'umap', 'pca'], default='tsne')
    ap.add_argument('--color', choices=['label', 'source'], default='label')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    d = np.load(args.z, allow_pickle=True)
    Z = d['z']
    print('z shape:', Z.shape)

    if args.method == 'tsne':
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, perplexity=min(30, len(Z) - 1), random_state=0).fit_transform(Z)
    elif args.method == 'umap':
        import umap
        proj = umap.UMAP(n_components=2, random_state=0).fit_transform(Z)
    else:
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2).fit_transform(Z)

    fig, ax = plt.subplots(figsize=(7, 6))

    color_key = args.color if args.color in d.files else None
    if color_key:
        labels = d[color_key]
        unique = list(dict.fromkeys(labels.tolist()))
        cmap = plt.get_cmap('tab10')
        for i, u in enumerate(unique):
            mask = labels == u
            ax.scatter(proj[mask, 0], proj[mask, 1], label=str(u), s=20, color=cmap(i % 10))
        ax.legend(loc='best', fontsize=8)
    else:
        ax.scatter(proj[:, 0], proj[:, 1], s=20)

    ax.set_title(f'Latent space ({args.method.upper()}, color={args.color})')
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150)
        print('Saved', args.out)
    plt.show()


if __name__ == '__main__':
    main()
