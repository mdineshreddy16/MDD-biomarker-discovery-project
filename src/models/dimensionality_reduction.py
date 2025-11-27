"""
Dimensionality reduction module for MDD biomarker discovery.
Implements PCA, t-SNE, UMAP, and Variational Autoencoder.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# UMAP (optional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class DimensionalityReducer:
    """
    Apply dimensionality reduction techniques.
    """
    
    def __init__(self, method: str = 'pca', **kwargs):
        """
        Initialize dimensionality reducer.
        
        Args:
            method: Reduction method ('pca', 'tsne', 'umap', 'vae')
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.model = None
        self.kwargs = kwargs
        
        if method == 'pca':
            n_components = kwargs.get('n_components', 0.95)
            self.model = PCA(n_components=n_components, **kwargs)
        elif method == 'tsne':
            self.model = TSNE(**kwargs)
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not installed. Install with: pip install umap-learn")
            self.model = umap.UMAP(**kwargs)
    
    def fit(self, X: np.ndarray) -> None:
        """Fit the reducer on data."""
        if self.method != 'vae':
            self.model.fit(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to lower dimensions."""
        if self.method == 'tsne':
            # t-SNE doesn't have separate fit/transform
            return self.model.fit_transform(X)
        else:
            return self.model.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        if self.method == 'vae':
            raise NotImplementedError("Use VAE class directly for VAE")
        return self.model.fit_transform(X)


class VAE(nn.Module):
    """
    Variational Autoencoder for learning latent depression representations.
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 16,
                 hidden_dims: list = [256, 128, 64]):
        """
        Initialize VAE.
        
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions
        """
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


class VAETrainer:
    """
    Train Variational Autoencoder.
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 16,
                 hidden_dims: list = [256, 128, 64],
                 learning_rate: float = 1e-3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize VAE trainer.
        
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            device: Device to train on
        """
        self.device = device
        self.model = VAE(input_dim, latent_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def loss_function(self,
                     reconstruction: torch.Tensor,
                     x: torch.Tensor,
                     mu: torch.Tensor,
                     logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            reconstruction: Reconstructed input
            x: Original input
            mu: Latent mean
            logvar: Latent log variance
            
        Returns:
            Total loss
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss
    
    def train_epoch(self, data_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in data_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            reconstruction, mu, logvar = self.model(batch)
            loss = self.loss_function(reconstruction, batch, mu, logvar)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader.dataset)
    
    def train(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """
        Train VAE on data.
        
        Args:
            X: Training data
            epochs: Number of epochs
            batch_size: Batch size
        """
        # Create data loader
        tensor_x = torch.FloatTensor(X)
        dataset = torch.utils.data.TensorDataset(tensor_x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            loss = self.train_epoch(loader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode data to latent space.
        
        Args:
            X: Input data
            
        Returns:
            Latent representations
        """
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.FloatTensor(X).to(self.device)
            mu, _ = self.model.encode(tensor_x)
            return mu.cpu().numpy()


if __name__ == "__main__":
    print("Dimensionality reduction module ready!")
    print("\nMethods available:")
    print("- PCA: Linear dimensionality reduction")
    print("- t-SNE: Nonlinear visualization")
    print("- UMAP: Fast nonlinear reduction")
    print("- VAE: Deep learning latent space")
