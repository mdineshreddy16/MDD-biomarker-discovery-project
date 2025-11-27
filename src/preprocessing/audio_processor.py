"""
Audio preprocessing module for MDD biomarker discovery.
Handles audio loading, cleaning, and basic feature extraction preparation.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class AudioProcessor:
    """
    Process audio files for depression biomarker analysis.
    Extracts acoustic features from speech recordings.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mfcc: int = 13,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128):
        """
        Initialize audio processor with configuration.
        
        Args:
            sample_rate: Target sampling rate in Hz
            n_mfcc: Number of MFCCs to extract
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of Mel bands
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
    def load_audio(self, 
                   file_path: str, 
                   duration: Optional[float] = None,
                   offset: float = 0.0) -> Tuple[np.ndarray, int]:
        """
        Load audio file with optional duration and offset.
        
        Args:
            file_path: Path to audio file
            duration: Maximum duration to load (in seconds)
            offset: Start reading after this time (in seconds)
            
        Returns:
            Tuple of (audio signal, sample rate)
        """
        try:
            audio, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                duration=duration,
                offset=offset,
                mono=True
            )
            return audio, sr
        except Exception as e:
            raise ValueError(f"Error loading audio file {file_path}: {e}")
    
    def remove_silence(self, 
                       audio: np.ndarray,
                       top_db: int = 20,
                       frame_length: int = 2048,
                       hop_length: int = 512) -> np.ndarray:
        """
        Remove silent parts from audio signal.
        
        Args:
            audio: Input audio signal
            top_db: Threshold below reference to consider silence
            frame_length: Frame length for silence detection
            hop_length: Hop length for silence detection
            
        Returns:
            Audio signal with silence removed
        """
        # Trim leading and trailing silence
        audio_trimmed, _ = librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        return audio_trimmed
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio signal to [-1, 1] range.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Normalized audio signal
        """
        return librosa.util.normalize(audio)
    
    def extract_segments(self,
                        audio: np.ndarray,
                        segment_duration: float = 3.0) -> List[np.ndarray]:
        """
        Split audio into fixed-length segments.
        
        Args:
            audio: Input audio signal
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of audio segments
        """
        segment_length = int(segment_duration * self.sample_rate)
        segments = []
        
        for start in range(0, len(audio), segment_length):
            end = start + segment_length
            segment = audio[start:end]
            
            # Pad last segment if needed
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)))
            
            segments.append(segment)
        
        return segments
    
    def preprocess_pipeline(self,
                           file_path: str,
                           remove_silence: bool = True,
                           normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        Complete preprocessing pipeline for audio file.
        
        Args:
            file_path: Path to audio file
            remove_silence: Whether to remove silent parts
            normalize: Whether to normalize amplitude
            
        Returns:
            Dictionary containing processed audio and metadata
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        
        # Remove silence
        if remove_silence:
            audio = self.remove_silence(audio)
        
        # Normalize
        if normalize:
            audio = self.normalize_audio(audio)
        
        # Calculate duration
        duration = len(audio) / sr
        
        return {
            'audio': audio,
            'sample_rate': sr,
            'duration': duration,
            'n_samples': len(audio)
        }
    
    def save_processed_audio(self, 
                            audio: np.ndarray,
                            output_path: str,
                            sample_rate: Optional[int] = None) -> None:
        """
        Save processed audio to file.
        
        Args:
            audio: Audio signal to save
            output_path: Output file path
            sample_rate: Sample rate (uses default if None)
        """
        sr = sample_rate if sample_rate else self.sample_rate
        sf.write(output_path, audio, sr)
        

class BatchAudioProcessor:
    """
    Process multiple audio files in batch.
    """
    
    def __init__(self, processor: AudioProcessor):
        """
        Initialize batch processor.
        
        Args:
            processor: AudioProcessor instance
        """
        self.processor = processor
    
    def process_directory(self,
                         input_dir: str,
                         output_dir: str,
                         file_pattern: str = "*.wav",
                         **kwargs) -> Dict[str, Dict]:
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_pattern: Glob pattern for audio files
            **kwargs: Additional arguments for preprocessing
            
        Returns:
            Dictionary mapping filenames to processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        audio_files = list(input_path.glob(file_pattern))
        
        print(f"Found {len(audio_files)} audio files to process")
        
        for i, audio_file in enumerate(audio_files):
            try:
                print(f"Processing {i+1}/{len(audio_files)}: {audio_file.name}")
                
                # Process audio
                processed = self.processor.preprocess_pipeline(
                    str(audio_file),
                    **kwargs
                )
                
                # Save processed audio
                output_file = output_path / audio_file.name
                self.processor.save_processed_audio(
                    processed['audio'],
                    str(output_file)
                )
                
                # Store metadata
                results[audio_file.stem] = {
                    'duration': processed['duration'],
                    'n_samples': processed['n_samples'],
                    'output_path': str(output_file)
                }
                
            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")
                results[audio_file.stem] = {'error': str(e)}
        
        return results


if __name__ == "__main__":
    # Example usage
    processor = AudioProcessor(sample_rate=16000)
    
    # Process single file
    # result = processor.preprocess_pipeline("path/to/audio.wav")
    # print(f"Processed audio: {result['duration']:.2f} seconds")
    
    # Batch processing
    # batch_processor = BatchAudioProcessor(processor)
    # results = batch_processor.process_directory(
    #     input_dir="data/raw/audio",
    #     output_dir="data/processed/audio",
    #     remove_silence=True,
    #     normalize=True
    # )
    
    print("Audio processor module ready!")
