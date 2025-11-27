"""
Multimodal feature fusion module.
Combines features from different modalities (audio, text, neuroimaging).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


class MultimodalFusion:
    """
    Combine features from multiple modalities.
    """
    
    def __init__(self,
                 fusion_strategy: str = 'concatenate',
                 normalize: bool = True,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize fusion module.
        
        Args:
            fusion_strategy: Strategy for combining features
                           ('concatenate', 'weighted', 'pca')
            normalize: Whether to normalize features before fusion
            weights: Modality weights for weighted fusion
        """
        self.fusion_strategy = fusion_strategy
        self.normalize = normalize
        self.weights = weights or {}
        
        # Scalers for each modality
        self.scalers = {}
        
    def add_scaler(self, modality: str, scaler_type: str = 'standard') -> None:
        """
        Add scaler for a specific modality.
        
        Args:
            modality: Name of modality (e.g., 'audio', 'text')
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        if scaler_type == 'standard':
            self.scalers[modality] = StandardScaler()
        elif scaler_type == 'minmax':
            self.scalers[modality] = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def fit_scalers(self, features_dict: Dict[str, np.ndarray]) -> None:
        """
        Fit scalers on training data.
        
        Args:
            features_dict: Dictionary mapping modality names to feature matrices
        """
        for modality, features in features_dict.items():
            if modality not in self.scalers:
                self.add_scaler(modality)
            
            self.scalers[modality].fit(features)
    
    def normalize_features(self, 
                          features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize features using fitted scalers.
        
        Args:
            features_dict: Dictionary of features
            
        Returns:
            Dictionary of normalized features
        """
        normalized = {}
        
        for modality, features in features_dict.items():
            if modality in self.scalers:
                normalized[modality] = self.scalers[modality].transform(features)
            else:
                normalized[modality] = features
        
        return normalized
    
    def concatenate_fusion(self, 
                          features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Simple concatenation of all features.
        
        Args:
            features_dict: Dictionary of features
            
        Returns:
            Concatenated feature matrix
        """
        feature_list = []
        
        for modality in sorted(features_dict.keys()):
            feature_list.append(features_dict[modality])
        
        return np.concatenate(feature_list, axis=1)
    
    def weighted_fusion(self, 
                       features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Weighted concatenation of features.
        
        Args:
            features_dict: Dictionary of features
            
        Returns:
            Weighted feature matrix
        """
        # Default equal weights if not specified
        if not self.weights:
            n_modalities = len(features_dict)
            self.weights = {mod: 1.0/n_modalities for mod in features_dict.keys()}
        
        # Apply weights and concatenate
        weighted_features = []
        
        for modality in sorted(features_dict.keys()):
            weight = self.weights.get(modality, 1.0)
            weighted_feat = features_dict[modality] * weight
            weighted_features.append(weighted_feat)
        
        return np.concatenate(weighted_features, axis=1)
    
    def pca_fusion(self,
                   features_dict: Dict[str, np.ndarray],
                   n_components: Optional[int] = None) -> np.ndarray:
        """
        Fuse features using PCA dimensionality reduction.
        
        Args:
            features_dict: Dictionary of features
            n_components: Number of PCA components
            
        Returns:
            PCA-transformed feature matrix
        """
        # First concatenate
        concatenated = self.concatenate_fusion(features_dict)
        
        # Apply PCA
        if n_components is None:
            n_components = min(50, concatenated.shape[1])
        
        pca = PCA(n_components=n_components)
        fused = pca.fit_transform(concatenated)
        
        return fused
    
    def fuse(self, 
             features_dict: Dict[str, np.ndarray],
             fit_scalers: bool = False) -> np.ndarray:
        """
        Fuse features using configured strategy.
        
        Args:
            features_dict: Dictionary mapping modality names to features
            fit_scalers: Whether to fit scalers (for training data)
            
        Returns:
            Fused feature matrix
        """
        # Normalize if requested
        if self.normalize:
            if fit_scalers:
                self.fit_scalers(features_dict)
            features_dict = self.normalize_features(features_dict)
        
        # Apply fusion strategy
        if self.fusion_strategy == 'concatenate':
            return self.concatenate_fusion(features_dict)
        elif self.fusion_strategy == 'weighted':
            return self.weighted_fusion(features_dict)
        elif self.fusion_strategy == 'pca':
            return self.pca_fusion(features_dict)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def create_feature_names(self, 
                            features_dict: Dict[str, List[str]]) -> List[str]:
        """
        Create feature names for fused features.
        
        Args:
            features_dict: Dictionary mapping modality to feature name lists
            
        Returns:
            List of feature names
        """
        all_names = []
        
        for modality in sorted(features_dict.keys()):
            names = features_dict[modality]
            prefixed_names = [f"{modality}_{name}" for name in names]
            all_names.extend(prefixed_names)
        
        return all_names


class FeatureSelector:
    """
    Select most important features from multimodal data.
    """
    
    def __init__(self, method: str = 'variance'):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('variance', 'correlation', 'mutual_info')
        """
        self.method = method
        self.selected_indices = None
    
    def select_by_variance(self, 
                          X: np.ndarray, 
                          threshold: float = 0.01) -> np.ndarray:
        """
        Select features with variance above threshold.
        
        Args:
            X: Feature matrix
            threshold: Minimum variance threshold
            
        Returns:
            Indices of selected features
        """
        variances = np.var(X, axis=0)
        selected = np.where(variances > threshold)[0]
        return selected
    
    def select_by_correlation(self,
                             X: np.ndarray,
                             y: np.ndarray,
                             n_features: int = 50) -> np.ndarray:
        """
        Select features most correlated with target.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_features: Number of features to select
            
        Returns:
            Indices of selected features
        """
        correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] 
                              for i in range(X.shape[1])])
        correlations = np.nan_to_num(correlations)
        
        selected = np.argsort(correlations)[-n_features:]
        return selected
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
            **kwargs) -> None:
        """
        Fit feature selector.
        
        Args:
            X: Feature matrix
            y: Target vector (optional, needed for some methods)
            **kwargs: Additional arguments for selection method
        """
        if self.method == 'variance':
            threshold = kwargs.get('threshold', 0.01)
            self.selected_indices = self.select_by_variance(X, threshold)
        elif self.method == 'correlation':
            if y is None:
                raise ValueError("Target y required for correlation selection")
            n_features = kwargs.get('n_features', 50)
            self.selected_indices = self.select_by_correlation(X, y, n_features)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using selected indices.
        
        Args:
            X: Feature matrix
            
        Returns:
            Selected features
        """
        if self.selected_indices is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        return X[:, self.selected_indices]
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                     **kwargs) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            **kwargs: Additional arguments
            
        Returns:
            Selected features
        """
        self.fit(X, y, **kwargs)
        return self.transform(X)


if __name__ == "__main__":
    print("Multimodal fusion module ready!")
    print("\nFusion strategies available:")
    print("- Concatenate: Simple feature concatenation")
    print("- Weighted: Weighted combination by modality")
    print("- PCA: Dimensionality reduction fusion")
    print("\nFeature selection methods:")
    print("- Variance: Remove low-variance features")
    print("- Correlation: Select features correlated with target")
