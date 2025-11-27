"""
Main pipeline for MDD Biomarker Discovery Project.
Orchestrates the complete workflow from data loading to clustering.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing import AudioProcessor, TextProcessor
from features import AudioFeatureExtractor, TextFeatureExtractor, MultimodalFusion
from models.dimensionality_reduction import DimensionalityReducer, VAETrainer
from models.clustering import ClusteringPipeline


class MDDPipeline:
    """
    Complete pipeline for MDD biomarker discovery.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data = {}
        self.features = {}
        self.reduced_features = {}
        self.clusters = {}
    
    def load_data(self):
        """Load raw data from configured paths."""
        print("=" * 50)
        print("STEP 1: Loading Data")
        print("=" * 50)
        
        dataset_config = self.config['dataset']
        raw_path = Path(dataset_config['raw_path'])
        
        # Check if data exists
        if not raw_path.exists():
            print(f"Data directory not found: {raw_path}")
            print("Please place your dataset in the data/raw directory")
            return False
        
        # Load metadata if available
        metadata_file = raw_path / 'metadata.csv'
        if metadata_file.exists():
            self.data['metadata'] = pd.read_csv(metadata_file)
            print(f"Loaded metadata: {len(self.data['metadata'])} entries")
        
        return True
    
    def preprocess_data(self):
        """Preprocess audio and text data."""
        print("\n" + "=" * 50)
        print("STEP 2: Preprocessing Data")
        print("=" * 50)
        
        # Audio preprocessing
        if self.config['audio']['features']:
            print("\nPreprocessing audio...")
            audio_processor = AudioProcessor(
                sample_rate=self.config['audio']['sample_rate']
            )
            # Process audio files here
            print("Audio preprocessing complete")
        
        # Text preprocessing
        if self.config['text']['features']:
            print("\nPreprocessing text...")
            text_processor = TextProcessor(
                lowercase=True,
                lemmatize=True
            )
            # Process text files here
            print("Text preprocessing complete")
    
    def extract_features(self):
        """Extract features from preprocessed data."""
        print("\n" + "=" * 50)
        print("STEP 3: Feature Extraction")
        print("=" * 50)
        
        feature_dict = {}
        
        # Extract audio features
        if 'audio' in self.data:
            print("\nExtracting audio features...")
            audio_extractor = AudioFeatureExtractor()
            # Extract features
            print("Audio features extracted")
        
        # Extract text features
        if 'text' in self.data:
            print("\nExtracting text features...")
            text_extractor = TextFeatureExtractor(
                use_tfidf=self.config['text']['use_tfidf']
            )
            # Extract features
            print("Text features extracted")
        
        # Fuse multimodal features
        if len(feature_dict) > 1:
            print("\nFusing multimodal features...")
            fusion = MultimodalFusion(
                fusion_strategy=self.config['features']['multimodal_fusion']
            )
            self.features['fused'] = fusion.fuse(feature_dict, fit_scalers=True)
            print(f"Fused features shape: {self.features['fused'].shape}")
    
    def reduce_dimensions(self):
        """Apply dimensionality reduction."""
        print("\n" + "=" * 50)
        print("STEP 4: Dimensionality Reduction")
        print("=" * 50)
        
        X = self.features.get('fused')
        if X is None:
            print("No features to reduce. Run feature extraction first.")
            return
        
        # PCA
        if self.config['dimensionality_reduction']['methods']['pca']['enabled']:
            print("\nApplying PCA...")
            pca_reducer = DimensionalityReducer(method='pca')
            self.reduced_features['pca'] = pca_reducer.fit_transform(X)
            print(f"PCA shape: {self.reduced_features['pca'].shape}")
        
        # t-SNE
        if self.config['dimensionality_reduction']['methods']['tsne']['enabled']:
            print("\nApplying t-SNE...")
            tsne_reducer = DimensionalityReducer(
                method='tsne',
                n_components=2,
                perplexity=30
            )
            self.reduced_features['tsne'] = tsne_reducer.fit_transform(X)
            print(f"t-SNE shape: {self.reduced_features['tsne'].shape}")
        
        # UMAP
        if self.config['dimensionality_reduction']['methods']['umap']['enabled']:
            print("\nApplying UMAP...")
            try:
                umap_reducer = DimensionalityReducer(
                    method='umap',
                    n_components=2
                )
                self.reduced_features['umap'] = umap_reducer.fit_transform(X)
                print(f"UMAP shape: {self.reduced_features['umap'].shape}")
            except ImportError:
                print("UMAP not installed. Skipping.")
    
    def perform_clustering(self):
        """Perform clustering to discover subtypes."""
        print("\n" + "=" * 50)
        print("STEP 5: Clustering")
        print("=" * 50)
        
        # Use PCA-reduced features for clustering
        X = self.reduced_features.get('pca')
        if X is None:
            print("No reduced features available. Using raw features.")
            X = self.features.get('fused')
        
        if X is None:
            print("No features available for clustering.")
            return
        
        # Initialize clustering pipeline
        pipeline = ClusteringPipeline()
        
        # Get enabled methods
        methods = []
        cluster_config = self.config['clustering']['methods']
        
        if cluster_config['kmeans']['enabled']:
            methods.append('kmeans')
        if cluster_config['gmm']['enabled']:
            methods.append('gmm')
        if cluster_config['spectral']['enabled']:
            methods.append('spectral')
        
        # Fit all methods
        results = pipeline.fit_all(X, methods=methods)
        
        # Find best clustering
        for method in methods:
            best_n, best_result = pipeline.get_best_clustering(method, metric='silhouette')
            print(f"\nBest {method.upper()}: {best_n} clusters")
            print(f"  Silhouette: {best_result['silhouette']:.3f}")
        
        self.clusters = results
    
    def save_results(self):
        """Save all results to output directory."""
        print("\n" + "=" * 50)
        print("STEP 6: Saving Results")
        print("=" * 50)
        
        output_path = Path(self.config['output']['results_path'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save reduced features
        for method, features in self.reduced_features.items():
            np.save(output_path / f'{method}_features.npy', features)
            print(f"Saved {method} features")
        
        # Save cluster labels
        for method, results in self.clusters.items():
            if method == 'hdbscan':
                labels = results.get('labels')
                if labels is not None:
                    np.save(output_path / f'{method}_labels.npy', labels)
            else:
                for n_clusters, result in results.items():
                    labels = result['labels']
                    np.save(output_path / f'{method}_{n_clusters}_labels.npy', labels)
        
        print(f"\nResults saved to {output_path}")
    
    def run(self, mode: str = 'full'):
        """
        Run the pipeline.
        
        Args:
            mode: Pipeline mode ('full', 'preprocess', 'features', 'clustering')
        """
        if mode in ['full', 'preprocess']:
            if not self.load_data():
                return
            self.preprocess_data()
        
        if mode in ['full', 'feature_extraction']:
            self.extract_features()
        
        if mode in ['full', 'dimensionality_reduction']:
            self.reduce_dimensions()
        
        if mode in ['full', 'clustering']:
            self.perform_clustering()
        
        if mode == 'full':
            self.save_results()
        
        print("\n" + "=" * 50)
        print("Pipeline Complete!")
        print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='MDD Biomarker Discovery Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--mode', default='full', 
                       choices=['full', 'preprocess', 'feature_extraction', 
                               'dimensionality_reduction', 'clustering'],
                       help='Pipeline mode')
    parser.add_argument('--dataset', default='daic-woz', help='Dataset name')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = MDDPipeline(config_path=args.config)
    pipeline.run(mode=args.mode)


if __name__ == "__main__":
    main()
