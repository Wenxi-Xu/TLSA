#!/usr/bin/env python3
"""
Lightweight visualization utilities.
Provides a t-SNE plot for clustering results (optionally after PCA).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize as l2_normalize

logger = logging.getLogger(__name__)

def visualize_kmeans_tsne(features: np.ndarray,
                          cluster_labels: np.ndarray,
                          save_path: str = None,
                          title: str = "K-means Clustering Results (t-SNE)",
                          pca_dim: int = 50,
                          perplexity: float = 30.0,
                          learning_rate: float = 200.0,
                          n_iter: int = 1000,
                          normalize_features: bool = True,
                          random_state: int = 42):
    """Visualize K-means clusters with t-SNE (optional PCA to `pca_dim`)."""
    logger.info("Generating t-SNE visualization for clustering...")

    X = features
    if normalize_features:
        X = l2_normalize(X)

    if pca_dim and X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=random_state).fit_transform(X)

    # Compatibility for different sklearn versions (fallback if needed)
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=float(learning_rate) if isinstance(learning_rate, (int, float, str)) else 200.0,
            n_iter=int(n_iter),
            init="pca",
            random_state=random_state,
            verbose=0,
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=float(learning_rate) if isinstance(learning_rate, (int, float, str)) else 200.0,
            init="pca",
            random_state=random_state,
            verbose=0,
        )
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(14, 11))
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.gist_ncar(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=[colors[i]],
            label=f'Cluster {label}',
            alpha=0.7,
            s=12,
            edgecolors='white',
            linewidth=0.2
        )

    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        plt.title(f'{title} ({len(unique_labels)} clusters)', fontsize=16)

    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"t-SNE visualization saved to {save_path}")

    plt.show()
    return plt.gcf()
