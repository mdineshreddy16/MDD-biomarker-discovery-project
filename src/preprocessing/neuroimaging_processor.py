"""
Neuroimaging preprocessing module for MDD biomarker discovery.
Handles fMRI and EEG data preprocessing.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Neuroimaging libraries (these are optional dependencies)
try:
    import nibabel as nib
    from nilearn import datasets, maskers, connectome
    from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
    from nilearn.image import mean_img, smooth_img
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("Warning: nilearn not installed. fMRI processing disabled.")

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: MNE not installed. EEG processing disabled.")


class fMRIProcessor:
    """
    Process fMRI (functional Magnetic Resonance Imaging) data.
    Extracts brain connectivity features.
    """
    
    def __init__(self,
                 atlas_name: str = 'harvard_oxford',
                 smoothing_fwhm: float = 6.0,
                 standardize: bool = True,
                 detrend: bool = True):
        """
        Initialize fMRI processor.
        
        Args:
            atlas_name: Brain atlas for parcellation
            smoothing_fwhm: Full-width half-maximum for smoothing
            standardize: Whether to standardize signals
            detrend: Whether to remove linear trends
        """
        if not NILEARN_AVAILABLE:
            raise ImportError("nilearn is required for fMRI processing. "
                            "Install with: pip install nilearn nibabel")
        
        self.atlas_name = atlas_name
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.atlas = None
        self.masker = None
        
    def load_atlas(self) -> None:
        """Load brain atlas for parcellation."""
        if self.atlas_name == 'harvard_oxford':
            self.atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        elif self.atlas_name == 'aal':
            self.atlas = datasets.fetch_atlas_aal()
        elif self.atlas_name == 'schaefer':
            self.atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
        else:
            raise ValueError(f"Unknown atlas: {self.atlas_name}")
        
        # Create masker
        self.masker = NiftiLabelsMasker(
            labels_img=self.atlas.maps,
            standardize=self.standardize,
            detrend=self.detrend,
            smoothing_fwhm=self.smoothing_fwhm
        )
    
    def load_fmri(self, file_path: str) -> nib.Nifti1Image:
        """
        Load fMRI data from NIfTI file.
        
        Args:
            file_path: Path to .nii or .nii.gz file
            
        Returns:
            Nibabel image object
        """
        return nib.load(file_path)
    
    def preprocess_fmri(self, img: nib.Nifti1Image) -> nib.Nifti1Image:
        """
        Apply basic preprocessing to fMRI image.
        
        Args:
            img: Input fMRI image
            
        Returns:
            Preprocessed image
        """
        # Smooth image
        if self.smoothing_fwhm > 0:
            img = smooth_img(img, self.smoothing_fwhm)
        
        return img
    
    def extract_time_series(self, img: nib.Nifti1Image) -> np.ndarray:
        """
        Extract ROI time series from fMRI data.
        
        Args:
            img: fMRI image
            
        Returns:
            Time series array (timepoints × ROIs)
        """
        if self.masker is None:
            self.load_atlas()
        
        time_series = self.masker.fit_transform(img)
        return time_series
    
    def compute_connectivity(self,
                           time_series: np.ndarray,
                           kind: str = 'correlation') -> np.ndarray:
        """
        Compute functional connectivity matrix.
        
        Args:
            time_series: Time series array (timepoints × ROIs)
            kind: Connectivity measure ('correlation', 'partial correlation', 'covariance')
            
        Returns:
            Connectivity matrix (ROIs × ROIs)
        """
        connectivity = connectome.ConnectivityMeasure(kind=kind)
        connectivity_matrix = connectivity.fit_transform([time_series])[0]
        return connectivity_matrix
    
    def extract_connectivity_features(self,
                                     connectivity_matrix: np.ndarray) -> Dict[str, float]:
        """
        Extract summary features from connectivity matrix.
        
        Args:
            connectivity_matrix: Connectivity matrix
            
        Returns:
            Dictionary of connectivity features
        """
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(connectivity_matrix, k=1)
        connectivity_values = connectivity_matrix[triu_indices]
        
        features = {
            'mean_connectivity': np.mean(connectivity_values),
            'std_connectivity': np.std(connectivity_values),
            'max_connectivity': np.max(connectivity_values),
            'min_connectivity': np.min(connectivity_values),
            'median_connectivity': np.median(connectivity_values),
            'positive_connections': np.sum(connectivity_values > 0),
            'negative_connections': np.sum(connectivity_values < 0),
            'strong_connections': np.sum(np.abs(connectivity_values) > 0.5)
        }
        
        return features
    
    def preprocess_pipeline(self, file_path: str) -> Dict:
        """
        Complete preprocessing pipeline for fMRI data.
        
        Args:
            file_path: Path to fMRI file
            
        Returns:
            Dictionary containing processed data and features
        """
        # Load and preprocess
        img = self.load_fmri(file_path)
        img = self.preprocess_fmri(img)
        
        # Extract time series
        time_series = self.extract_time_series(img)
        
        # Compute connectivity
        connectivity = self.compute_connectivity(time_series)
        
        # Extract features
        features = self.extract_connectivity_features(connectivity)
        
        return {
            'time_series': time_series,
            'connectivity_matrix': connectivity,
            'features': features,
            'n_rois': time_series.shape[1],
            'n_timepoints': time_series.shape[0]
        }


class EEGProcessor:
    """
    Process EEG (Electroencephalography) data.
    Extracts neural oscillation and connectivity features.
    """
    
    def __init__(self,
                 sampling_rate: float = 250.0,
                 l_freq: float = 0.5,
                 h_freq: float = 50.0):
        """
        Initialize EEG processor.
        
        Args:
            sampling_rate: Sampling frequency in Hz
            l_freq: Low cutoff frequency for bandpass filter
            h_freq: High cutoff frequency for bandpass filter
        """
        if not MNE_AVAILABLE:
            raise ImportError("MNE is required for EEG processing. "
                            "Install with: pip install mne")
        
        self.sampling_rate = sampling_rate
        self.l_freq = l_freq
        self.h_freq = h_freq
    
    def load_eeg(self, file_path: str) -> mne.io.Raw:
        """
        Load EEG data from file.
        
        Args:
            file_path: Path to EEG file (supports various formats)
            
        Returns:
            MNE Raw object
        """
        # Auto-detect file format
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.fif':
            raw = mne.io.read_raw_fif(file_path, preload=True)
        elif file_ext == '.edf':
            raw = mne.io.read_raw_edf(file_path, preload=True)
        elif file_ext == '.set':
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return raw
    
    def preprocess_eeg(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply basic preprocessing to EEG data.
        
        Args:
            raw: Raw EEG data
            
        Returns:
            Preprocessed EEG data
        """
        # Apply bandpass filter
        raw_filtered = raw.copy().filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            fir_design='firwin'
        )
        
        # Remove bad channels (simple threshold-based)
        # In practice, use more sophisticated artifact rejection
        raw_filtered.interpolate_bads()
        
        return raw_filtered
    
    def extract_band_powers(self, raw: mne.io.Raw) -> Dict[str, np.ndarray]:
        """
        Extract power in different frequency bands.
        
        Args:
            raw: Preprocessed EEG data
            
        Returns:
            Dictionary of band powers
        """
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        band_powers = {}
        
        for band_name, (fmin, fmax) in bands.items():
            # Filter to band
            raw_band = raw.copy().filter(fmin, fmax)
            
            # Compute power
            data = raw_band.get_data()
            power = np.mean(data ** 2, axis=1)
            
            band_powers[band_name] = power
        
        return band_powers
    
    def extract_features(self, raw: mne.io.Raw) -> Dict[str, float]:
        """
        Extract summary EEG features.
        
        Args:
            raw: Preprocessed EEG data
            
        Returns:
            Dictionary of EEG features
        """
        # Extract band powers
        band_powers = self.extract_band_powers(raw)
        
        features = {}
        
        # Average power in each band
        for band_name, powers in band_powers.items():
            features[f'{band_name}_power_mean'] = np.mean(powers)
            features[f'{band_name}_power_std'] = np.std(powers)
        
        # Band ratios (common depression markers)
        alpha_mean = np.mean(band_powers['alpha'])
        theta_mean = np.mean(band_powers['theta'])
        beta_mean = np.mean(band_powers['beta'])
        
        features['theta_alpha_ratio'] = theta_mean / alpha_mean if alpha_mean > 0 else 0
        features['theta_beta_ratio'] = theta_mean / beta_mean if beta_mean > 0 else 0
        
        return features
    
    def preprocess_pipeline(self, file_path: str) -> Dict:
        """
        Complete preprocessing pipeline for EEG data.
        
        Args:
            file_path: Path to EEG file
            
        Returns:
            Dictionary containing processed data and features
        """
        # Load and preprocess
        raw = self.load_eeg(file_path)
        raw = self.preprocess_eeg(raw)
        
        # Extract features
        features = self.extract_features(raw)
        
        return {
            'raw': raw,
            'features': features,
            'n_channels': len(raw.ch_names),
            'duration': raw.times[-1]
        }


if __name__ == "__main__":
    print("Neuroimaging processor module ready!")
    print(f"fMRI processing available: {NILEARN_AVAILABLE}")
    print(f"EEG processing available: {MNE_AVAILABLE}")
