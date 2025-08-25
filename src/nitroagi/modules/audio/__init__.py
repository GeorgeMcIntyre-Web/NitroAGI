"""
Audio Module for NitroAGI NEXUS
Provides audio processing, speech-to-text, and text-to-speech capabilities
"""

from .audio_module import AudioModule
from .processors import (
    AudioProcessor,
    SpeechToTextProcessor,
    TextToSpeechProcessor,
    AudioAnalyzer
)

__all__ = [
    "AudioModule",
    "AudioProcessor",
    "SpeechToTextProcessor", 
    "TextToSpeechProcessor",
    "AudioAnalyzer"
]