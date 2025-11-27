"""
Models package for MDD Biomarker Discovery.
Contains dimensionality reduction and clustering implementations.
"""

from .dimensionality_reduction import DimensionalityReducer, VAE, VAETrainer
from .clustering import ClusteringPipeline

__all__ = [
    'DimensionalityReducer',
    'VAE',
    'VAETrainer',
    'ClusteringPipeline'
]
