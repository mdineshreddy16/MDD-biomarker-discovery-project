"""
Preprocessing package initialization.
"""

from .audio_processor import AudioProcessor, BatchAudioProcessor
from .text_processor import TextProcessor, BatchTextProcessor

# Conditional imports for neuroimaging
try:
    from .neuroimaging_processor import fMRIProcessor, EEGProcessor
    NEUROIMAGING_AVAILABLE = True
except ImportError:
    NEUROIMAGING_AVAILABLE = False
    fMRIProcessor = None
    EEGProcessor = None

__all__ = [
    'AudioProcessor',
    'BatchAudioProcessor',
    'TextProcessor',
    'BatchTextProcessor',
    'fMRIProcessor',
    'EEGProcessor',
    'NEUROIMAGING_AVAILABLE'
]
