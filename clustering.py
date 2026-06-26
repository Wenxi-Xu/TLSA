import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple, Optional
import logging
logger = logging.getLogger(__name__)

class ClusterManager:
    """Clustering manager."""
    
    def __init__(self, args):
        self.args = args

        self.num_clusters = None
        self.cluster_centers = None
        self.cluster_labels = None
        
    def perform_clustering(self, features: np.ndarray, num_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform K-means clustering."""

        self.num_clusters = num_clusters
        
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=self.args.seed,
            n_init=10
        )
        
        cluster_labels = kmeans.fit_predict(features)
        cluster_centers = kmeans.cluster_centers_
        
        if len(np.unique(cluster_labels)) > 1:
            silhouette = silhouette_score(features, cluster_labels)
            logger.info(f"Clustering completed: {num_clusters} clusters, silhouette: {silhouette:.4f}")
        else:
            logger.warning("Only one cluster found!")
        
        self.cluster_labels = cluster_labels
        self.cluster_centers = cluster_centers
        
        return cluster_labels, cluster_centers
    

def create_cluster_manager(args) -> ClusterManager:
    """Create clustering manager."""
    return ClusterManager(args)