"""
Clustering module for discovering depression subtypes.
Implements K-Means, GMM, Spectral Clustering, and HDBSCAN.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Optional HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class ClusteringPipeline:
    """
    Comprehensive clustering pipeline with multiple algorithms.
    """
    
    def __init__(self):
        """Initialize clustering pipeline."""
        self.models = {}
        self.results = {}
    
    def fit_kmeans(self,
                   X: np.ndarray,
                   n_clusters_range: List[int] = [2, 3, 4, 5, 6],
                   **kwargs) -> Dict:
        """
        Fit K-Means with multiple cluster counts.
        
        Args:
            X: Feature matrix
            n_clusters_range: List of cluster counts to try
            **kwargs: Additional K-Means parameters
            
        Returns:
            Dictionary of results for each n_clusters
        """
        results = {}
        
        for n_clusters in n_clusters_range:
            print(f"Fitting K-Means with {n_clusters} clusters...")
            
            model = KMeans(
                n_clusters=n_clusters,
                n_init=50,
                max_iter=300,
                random_state=42,
                **kwargs
            )
            
            labels = model.fit_predict(X)
            
            # Compute metrics
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            
            results[n_clusters] = {
                'model': model,
                'labels': labels,
                'inertia': model.inertia_,
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski_harabasz
            }
            
            print(f"  Silhouette: {silhouette:.3f}")
            print(f"  Davies-Bouldin: {davies_bouldin:.3f}")
        
        self.models['kmeans'] = results
        return results
    
    def fit_gmm(self,
                X: np.ndarray,
                n_components_range: List[int] = [2, 3, 4, 5, 6],
                **kwargs) -> Dict:
        """
        Fit Gaussian Mixture Model with multiple component counts.
        
        Args:
            X: Feature matrix
            n_components_range: List of component counts to try
            **kwargs: Additional GMM parameters
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        for n_components in n_components_range:
            print(f"Fitting GMM with {n_components} components...")
            
            model = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                n_init=10,
                random_state=42,
                **kwargs
            )
            
            model.fit(X)
            labels = model.predict(X)
            
            # Compute metrics
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            
            results[n_components] = {
                'model': model,
                'labels': labels,
                'bic': model.bic(X),
                'aic': model.aic(X),
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski_harabasz
            }
            
            print(f"  BIC: {model.bic(X):.2f}")
            print(f"  Silhouette: {silhouette:.3f}")
        
        self.models['gmm'] = results
        return results
    
    def fit_spectral(self,
                    X: np.ndarray,
                    n_clusters_range: List[int] = [2, 3, 4, 5, 6],
                    **kwargs) -> Dict:
        """
        Fit Spectral Clustering with multiple cluster counts.
        
        Args:
            X: Feature matrix
            n_clusters_range: List of cluster counts to try
            **kwargs: Additional Spectral parameters
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        for n_clusters in n_clusters_range:
            print(f"Fitting Spectral Clustering with {n_clusters} clusters...")
            
            model = SpectralClustering(
                n_clusters=n_clusters,
                affinity='rbf',
                random_state=42,
                **kwargs
            )
            
            labels = model.fit_predict(X)
            
            # Compute metrics
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            
            results[n_clusters] = {
                'model': model,
                'labels': labels,
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski_harabasz
            }
            
            print(f"  Silhouette: {silhouette:.3f}")
        
        self.models['spectral'] = results
        return results
    
    def fit_hdbscan(self, X: np.ndarray, **kwargs) -> Dict:
        """
        Fit HDBSCAN (density-based clustering).
        
        Args:
            X: Feature matrix
            **kwargs: HDBSCAN parameters
            
        Returns:
            Dictionary of results
        """
        if not HDBSCAN_AVAILABLE:
            print("HDBSCAN not available. Install with: pip install hdbscan")
            return {}
        
        print("Fitting HDBSCAN...")
        
        model = hdbscan.HDBSCAN(
            min_cluster_size=kwargs.get('min_cluster_size', 5),
            min_samples=kwargs.get('min_samples', 3),
            **kwargs
        )
        
        labels = model.fit_predict(X)
        
        # Filter out noise points (-1 label) for metrics
        mask = labels != -1
        if np.sum(mask) < 2:
            print("  Too few non-noise points for evaluation")
            return {}
        
        X_filtered = X[mask]
        labels_filtered = labels[mask]
        
        n_clusters = len(set(labels_filtered))
        
        results = {
            'model': model,
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': np.sum(labels == -1)
        }
        
        if n_clusters > 1:
            results['silhouette'] = silhouette_score(X_filtered, labels_filtered)
            results['davies_bouldin'] = davies_bouldin_score(X_filtered, labels_filtered)
            results['calinski_harabasz'] = calinski_harabasz_score(X_filtered, labels_filtered)
            
            print(f"  Found {n_clusters} clusters ({results['n_noise']} noise points)")
            print(f"  Silhouette: {results['silhouette']:.3f}")
        
        self.models['hdbscan'] = results
        return results
    
    def fit_all(self,
                X: np.ndarray,
                methods: List[str] = ['kmeans', 'gmm', 'spectral'],
                **kwargs) -> Dict:
        """
        Fit all specified clustering methods.
        
        Args:
            X: Feature matrix
            methods: List of methods to fit
            **kwargs: Parameters for each method
            
        Returns:
            Dictionary of all results
        """
        if 'kmeans' in methods:
            self.fit_kmeans(X, **kwargs.get('kmeans', {}))
        
        if 'gmm' in methods:
            self.fit_gmm(X, **kwargs.get('gmm', {}))
        
        if 'spectral' in methods:
            self.fit_spectral(X, **kwargs.get('spectral', {}))
        
        if 'hdbscan' in methods:
            self.fit_hdbscan(X, **kwargs.get('hdbscan', {}))
        
        return self.models
    
    def get_best_clustering(self, 
                           method: str = 'kmeans',
                           metric: str = 'silhouette') -> Tuple[int, Dict]:
        """
        Get best clustering result for a method.
        
        Args:
            method: Clustering method
            metric: Metric to optimize ('silhouette', 'davies_bouldin', 'calinski_harabasz')
            
        Returns:
            Tuple of (best_n_clusters, results_dict)
        """
        if method not in self.models:
            raise ValueError(f"Method {method} not fitted yet")
        
        results = self.models[method]
        
        if method == 'hdbscan':
            return results.get('n_clusters', 0), results
        
        # For other methods, find best by metric
        best_n = None
        best_score = float('-inf') if metric != 'davies_bouldin' else float('inf')
        
        for n, res in results.items():
            score = res.get(metric, 0)
            
            if metric == 'davies_bouldin':
                # Lower is better
                if score < best_score:
                    best_score = score
                    best_n = n
            else:
                # Higher is better
                if score > best_score:
                    best_score = score
                    best_n = n
        
        return best_n, results[best_n]


if __name__ == "__main__":
    print("Clustering module ready!")
    print("\nAlgorithms available:")
    print("- K-Means: Centroid-based clustering")
    print("- GMM: Probabilistic soft clustering")
    print("- Spectral: Graph-based clustering")
    print("- HDBSCAN: Density-based clustering (optional)")
