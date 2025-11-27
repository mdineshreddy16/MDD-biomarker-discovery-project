"""
Audio feature extraction module for depression biomarker discovery.
Extracts comprehensive acoustic features from speech.
"""

import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """
    Extract acoustic features relevant to depression detection.
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 n_mfcc: int = 13,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Audio sampling rate
            n_mfcc: Number of MFCCs to extract
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of Mel bands
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def extract_mfcc(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract Mel-Frequency Cepstral Coefficients.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with MFCC features and statistics
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Compute delta and delta-delta
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        return {
            'mfcc': mfcc,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2,
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_std': np.std(mfcc, axis=1),
            'mfcc_delta_mean': np.mean(mfcc_delta, axis=1),
            'mfcc_delta2_mean': np.mean(mfcc_delta2, axis=1)
        }
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features from audio.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of spectral features
        """
        # Spectral centroid
        spec_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        spec_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        # Spectral contrast
        spec_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Spectral flatness
        spec_flatness = librosa.feature.spectral_flatness(
            y=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        return {
            'spectral_centroid_mean': np.mean(spec_centroid),
            'spectral_centroid_std': np.std(spec_centroid),
            'spectral_rolloff_mean': np.mean(spec_rolloff),
            'spectral_rolloff_std': np.std(spec_rolloff),
            'spectral_bandwidth_mean': np.mean(spec_bandwidth),
            'spectral_bandwidth_std': np.std(spec_bandwidth),
            'spectral_contrast_mean': np.mean(spec_contrast, axis=1),
            'spectral_flatness_mean': np.mean(spec_flatness),
            'spectral_flatness_std': np.std(spec_flatness)
        }
    
    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract prosodic features (pitch, energy, rhythm).
        Critical for depression detection.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of prosodic features
        """
        # Fundamental frequency (pitch)
        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        # Remove invalid values
        f0_valid = f0[f0 > 0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        # RMS energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            tempo = float(tempo)
        except:
            tempo = 0.0
        
        features = {
            'pitch_mean': np.mean(f0_valid) if len(f0_valid) > 0 else 0,
            'pitch_std': np.std(f0_valid) if len(f0_valid) > 0 else 0,
            'pitch_min': np.min(f0_valid) if len(f0_valid) > 0 else 0,
            'pitch_max': np.max(f0_valid) if len(f0_valid) > 0 else 0,
            'pitch_range': np.ptp(f0_valid) if len(f0_valid) > 0 else 0,
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'energy_mean': np.mean(rms),
            'energy_std': np.std(rms),
            'tempo': tempo
        }
        
        return features
    
    def extract_chroma_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract chromagram features.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of chroma features
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return {
            'chroma_mean': np.mean(chroma, axis=1),
            'chroma_std': np.std(chroma, axis=1)
        }
    
    def extract_rhythm_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract rhythm and temporal features.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of rhythm features
        """
        # Onset strength
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate
        )
        
        # Tempogram
        try:
            tempogram = librosa.feature.tempogram(
                onset_envelope=onset_env,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            tempogram_mean = np.mean(tempogram)
            tempogram_std = np.std(tempogram)
        except:
            tempogram_mean = 0.0
            tempogram_std = 0.0
        
        return {
            'onset_strength_mean': np.mean(onset_env),
            'onset_strength_std': np.std(onset_env),
            'tempogram_mean': tempogram_mean,
            'tempogram_std': tempogram_std
        }
    
    def extract_pause_features(self, audio: np.ndarray, 
                              energy_threshold: float = 0.01) -> Dict[str, float]:
        """
        Extract pause/silence features (important for depression).
        
        Args:
            audio: Audio signal
            energy_threshold: Threshold to detect pauses
            
        Returns:
            Dictionary of pause features
        """
        # Compute frame energy
        frame_length = self.n_fft
        hop_length = self.hop_length
        
        frames = librosa.util.frame(audio, frame_length=frame_length, 
                                    hop_length=hop_length)
        energy = np.sum(frames**2, axis=0)
        
        # Normalize energy
        energy = energy / np.max(energy) if np.max(energy) > 0 else energy
        
        # Detect pauses
        is_pause = energy < energy_threshold
        
        # Count and measure pauses
        pause_changes = np.diff(is_pause.astype(int))
        n_pauses = np.sum(pause_changes == 1)
        
        # Pause durations
        pause_durations = []
        in_pause = False
        pause_start = 0
        
        for i, is_p in enumerate(is_pause):
            if is_p and not in_pause:
                pause_start = i
                in_pause = True
            elif not is_p and in_pause:
                pause_durations.append(i - pause_start)
                in_pause = False
        
        total_frames = len(is_pause)
        pause_frames = np.sum(is_pause)
        
        return {
            'n_pauses': n_pauses,
            'pause_ratio': pause_frames / total_frames if total_frames > 0 else 0,
            'avg_pause_duration': np.mean(pause_durations) if pause_durations else 0,
            'max_pause_duration': np.max(pause_durations) if pause_durations else 0,
            'speech_ratio': 1 - (pause_frames / total_frames) if total_frames > 0 else 0
        }
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all acoustic features.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary containing all features
        """
        features = {}
        
        # MFCC features
        mfcc_feats = self.extract_mfcc(audio)
        features.update({f'mfcc_{k}': v for k, v in mfcc_feats.items()})
        
        # Spectral features
        spectral_feats = self.extract_spectral_features(audio)
        features.update(spectral_feats)
        
        # Prosodic features
        prosodic_feats = self.extract_prosodic_features(audio)
        features.update(prosodic_feats)
        
        # Chroma features
        chroma_feats = self.extract_chroma_features(audio)
        features.update({f'chroma_{k}': v for k, v in chroma_feats.items()})
        
        # Rhythm features
        rhythm_feats = self.extract_rhythm_features(audio)
        features.update(rhythm_feats)
        
        # Pause features
        pause_feats = self.extract_pause_features(audio)
        features.update(pause_feats)
        
        return features
    
    def flatten_features(self, features: Dict) -> np.ndarray:
        """
        Flatten feature dictionary into a single vector.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Flattened feature vector
        """
        flat_features = []
        
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                if value.ndim > 1:
                    # Skip time-series features, use only statistics
                    continue
                flat_features.extend(value.flatten())
            else:
                flat_features.append(value)
        
        return np.array(flat_features)


if __name__ == "__main__":
    print("Audio feature extraction module ready!")
    print("\nFeatures extracted:")
    print("- MFCC (13 coefficients + deltas)")
    print("- Spectral (centroid, rolloff, bandwidth, contrast, flatness)")
    print("- Prosodic (pitch, energy, tempo)")
    print("- Chroma (12 pitch classes)")
    print("- Rhythm (onset strength, tempogram)")
    print("- Pause (speech/silence ratio, pause duration)")
