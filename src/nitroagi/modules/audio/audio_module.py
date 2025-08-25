"""
Audio Module Implementation for NitroAGI NEXUS
Handles audio processing, speech recognition, and synthesis
"""

import asyncio
import base64
import io
import logging
import wave
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from nitroagi.core.base import (
    AIModule,
    ModuleCapability,
    ModuleRequest,
    ModuleResponse,
    ProcessingContext
)
from nitroagi.core.exceptions import ModuleException
from nitroagi.utils.logging import get_logger

# Import processors
from .processors import (
    AudioProcessor,
    SpeechToTextProcessor,
    TextToSpeechProcessor,
    AudioAnalyzer
)


class AudioTask(Enum):
    """Types of audio tasks."""
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO_CLASSIFICATION = "audio_classification"
    SPEAKER_RECOGNITION = "speaker_recognition"
    EMOTION_DETECTION = "emotion_detection"
    NOISE_REDUCTION = "noise_reduction"
    AUDIO_ENHANCEMENT = "audio_enhancement"
    TRANSCRIPTION = "transcription"


@dataclass
class AudioResult:
    """Result from audio processing."""
    task: AudioTask
    text: Optional[str] = None
    audio_data: Optional[bytes] = None
    classifications: Optional[Dict[str, float]] = None
    speaker_id: Optional[str] = None
    emotions: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    sample_rate: int = 16000
    duration_seconds: float = 0.0


class AudioModule(AIModule):
    """
    Audio module for NitroAGI NEXUS.
    Provides comprehensive audio processing capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the audio module.
        
        Args:
            config: Module configuration
        """
        super().__init__(config)
        self.logger = get_logger(__name__)
        
        # Initialize processors
        self.audio_processor = AudioProcessor()
        self.speech_to_text = SpeechToTextProcessor()
        self.text_to_speech = TextToSpeechProcessor()
        self.audio_analyzer = AudioAnalyzer()
        
        # Configuration
        self.sample_rate = config.get("sample_rate", 16000)
        self.max_duration = config.get("max_duration_seconds", 300)  # 5 minutes
        self.enable_gpu = config.get("enable_gpu", False)
        self.language = config.get("language", "en")
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        
        # Cache for processed audio
        self.audio_cache = {}
        self.cache_size = config.get("cache_size", 50)
    
    async def initialize(self) -> bool:
        """
        Initialize the audio module and load models.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing Audio Module...")
            
            # Initialize processors
            await self.audio_processor.initialize(sample_rate=self.sample_rate)
            await self.speech_to_text.initialize(
                language=self.language,
                use_gpu=self.enable_gpu
            )
            await self.text_to_speech.initialize(
                language=self.language,
                sample_rate=self.sample_rate
            )
            await self.audio_analyzer.initialize()
            
            self._initialized = True
            self.logger.info("Audio Module initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Audio Module: {e}")
            self._initialized = False
            return False
    
    async def process(self, request: ModuleRequest) -> ModuleResponse:
        """
        Process an audio request.
        
        Args:
            request: The audio processing request
            
        Returns:
            ModuleResponse with audio results
        """
        if not self._initialized:
            return ModuleResponse(
                request_id=request.id,
                module_name=self.name,
                status="error",
                error="Audio module not initialized"
            )
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Determine task type
            task = self._determine_task(request)
            
            # Process based on task
            if task == AudioTask.TEXT_TO_SPEECH:
                # Extract text for TTS
                text = self._extract_text(request.data)
                result = await self._process_text_to_speech(text, request)
            else:
                # Extract audio for processing
                audio_data = await self._extract_audio(request.data)
                result = await self._process_audio_task(audio_data, task, request)
            
            # Build response
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return ModuleResponse(
                request_id=request.id,
                module_name=self.name,
                status="success",
                data=self._format_result(result),
                processing_time_ms=processing_time,
                confidence_score=result.confidence
            )
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}", exc_info=True)
            return ModuleResponse(
                request_id=request.id,
                module_name=self.name,
                status="error",
                error=str(e),
                processing_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    async def _extract_audio(self, data: Any) -> np.ndarray:
        """
        Extract audio from request data.
        
        Args:
            data: Request data containing audio
            
        Returns:
            Audio as numpy array
        """
        if isinstance(data, str):
            # Check if it's a file path
            if data.startswith('/') or data.startswith('C:\\'):
                return await self._load_audio_file(data)
            # Check if it's base64 encoded
            elif data.startswith('data:audio'):
                # Extract base64 data
                base64_str = data.split(',')[1] if ',' in data else data
                audio_bytes = base64.b64decode(base64_str)
                return await self._decode_audio_bytes(audio_bytes)
            else:
                raise ValueError(f"Unknown string format for audio data")
        
        elif isinstance(data, bytes):
            # Raw audio bytes
            return await self._decode_audio_bytes(data)
        
        elif isinstance(data, np.ndarray):
            # Already a numpy array
            return data
        
        elif isinstance(data, dict):
            # Extract audio from dict
            if "audio" in data:
                return await self._extract_audio(data["audio"])
            elif "audio_path" in data:
                return await self._load_audio_file(data["audio_path"])
            elif "audio_url" in data:
                # Download audio from URL
                return await self._download_audio(data["audio_url"])
            elif "text" in data:
                # This might be for TTS
                raise ValueError("Text data provided for audio extraction")
        
        raise ValueError(f"Cannot extract audio from data type: {type(data)}")
    
    def _extract_text(self, data: Any) -> str:
        """
        Extract text from request data for TTS.
        
        Args:
            data: Request data containing text
            
        Returns:
            Text string
        """
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            if "text" in data:
                return data["text"]
            elif "message" in data:
                return data["message"]
        
        raise ValueError(f"Cannot extract text from data type: {type(data)}")
    
    def _determine_task(self, request: ModuleRequest) -> AudioTask:
        """
        Determine the audio task from request.
        
        Args:
            request: Module request
            
        Returns:
            AudioTask enum value
        """
        # Check capabilities
        if ModuleCapability.SPEECH_TO_TEXT in request.required_capabilities:
            return AudioTask.SPEECH_TO_TEXT
        elif ModuleCapability.TEXT_TO_SPEECH in request.required_capabilities:
            return AudioTask.TEXT_TO_SPEECH
        
        # Check explicit task in data
        if isinstance(request.data, dict):
            task_str = request.data.get("task", "").lower()
            
            if "speech" in task_str and "text" in task_str:
                if "to_text" in task_str or "transcri" in task_str:
                    return AudioTask.SPEECH_TO_TEXT
                else:
                    return AudioTask.TEXT_TO_SPEECH
            elif "transcri" in task_str:
                return AudioTask.TRANSCRIPTION
            elif "emotion" in task_str:
                return AudioTask.EMOTION_DETECTION
            elif "speaker" in task_str:
                return AudioTask.SPEAKER_RECOGNITION
            elif "noise" in task_str:
                return AudioTask.NOISE_REDUCTION
            elif "enhance" in task_str:
                return AudioTask.AUDIO_ENHANCEMENT
            elif "classif" in task_str:
                return AudioTask.AUDIO_CLASSIFICATION
            
            # Check if text is provided (likely TTS)
            if "text" in request.data and "audio" not in request.data:
                return AudioTask.TEXT_TO_SPEECH
        
        # Default to speech-to-text
        return AudioTask.SPEECH_TO_TEXT
    
    async def _process_audio_task(
        self,
        audio_data: np.ndarray,
        task: AudioTask,
        request: ModuleRequest
    ) -> AudioResult:
        """
        Process specific audio task.
        
        Args:
            audio_data: Input audio as numpy array
            task: Audio task to perform
            request: Original request
            
        Returns:
            AudioResult with task results
        """
        # Preprocess audio
        processed_audio = await self.audio_processor.preprocess(
            audio_data,
            target_sample_rate=self.sample_rate
        )
        
        # Calculate duration
        duration = len(processed_audio) / self.sample_rate
        
        result = AudioResult(
            task=task,
            metadata={},
            sample_rate=self.sample_rate,
            duration_seconds=duration
        )
        
        if task == AudioTask.SPEECH_TO_TEXT:
            # Convert speech to text
            transcription = await self.speech_to_text.transcribe(
                processed_audio,
                language=self.language
            )
            result.text = transcription["text"]
            result.confidence = transcription.get("confidence", 0.9)
            result.metadata["words"] = transcription.get("words", [])
            
        elif task == AudioTask.TRANSCRIPTION:
            # Detailed transcription with timestamps
            transcription = await self.speech_to_text.transcribe_detailed(
                processed_audio,
                language=self.language
            )
            result.text = transcription["text"]
            result.metadata["segments"] = transcription.get("segments", [])
            result.confidence = transcription.get("confidence", 0.9)
            
        elif task == AudioTask.EMOTION_DETECTION:
            # Detect emotions in audio
            emotions = await self.audio_analyzer.detect_emotions(processed_audio)
            result.emotions = emotions
            result.confidence = max(emotions.values()) if emotions else 0.0
            
        elif task == AudioTask.SPEAKER_RECOGNITION:
            # Identify speaker
            speaker_data = await self.audio_analyzer.identify_speaker(processed_audio)
            result.speaker_id = speaker_data["speaker_id"]
            result.confidence = speaker_data.get("confidence", 0.8)
            result.metadata["embeddings"] = speaker_data.get("embeddings")
            
        elif task == AudioTask.AUDIO_CLASSIFICATION:
            # Classify audio
            classifications = await self.audio_analyzer.classify_audio(processed_audio)
            result.classifications = classifications
            result.confidence = max(classifications.values()) if classifications else 0.0
            
        elif task == AudioTask.NOISE_REDUCTION:
            # Reduce noise in audio
            cleaned_audio = await self.audio_processor.reduce_noise(processed_audio)
            result.audio_data = self._encode_audio(cleaned_audio)
            result.confidence = 0.95
            
        elif task == AudioTask.AUDIO_ENHANCEMENT:
            # Enhance audio quality
            enhanced_audio = await self.audio_processor.enhance_audio(processed_audio)
            result.audio_data = self._encode_audio(enhanced_audio)
            result.confidence = 0.95
        
        return result
    
    async def _process_text_to_speech(
        self,
        text: str,
        request: ModuleRequest
    ) -> AudioResult:
        """
        Process text-to-speech request.
        
        Args:
            text: Text to convert to speech
            request: Original request
            
        Returns:
            AudioResult with synthesized speech
        """
        # Get voice settings from request if available
        voice_settings = {}
        if isinstance(request.data, dict):
            voice_settings = {
                "voice": request.data.get("voice", "default"),
                "speed": request.data.get("speed", 1.0),
                "pitch": request.data.get("pitch", 1.0),
                "emotion": request.data.get("emotion", "neutral")
            }
        
        # Synthesize speech
        audio_data = await self.text_to_speech.synthesize(
            text,
            language=self.language,
            **voice_settings
        )
        
        # Calculate duration
        duration = len(audio_data) / self.sample_rate
        
        result = AudioResult(
            task=AudioTask.TEXT_TO_SPEECH,
            text=text,
            audio_data=self._encode_audio(audio_data),
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            confidence=0.95,
            metadata=voice_settings
        )
        
        return result
    
    async def _load_audio_file(self, file_path: str) -> np.ndarray:
        """
        Load audio from file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio as numpy array
        """
        import soundfile as sf
        
        # Read audio file
        audio_data, sample_rate = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio_data = await self.audio_processor.resample(
                audio_data,
                original_rate=sample_rate,
                target_rate=self.sample_rate
            )
        
        return audio_data
    
    async def _decode_audio_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """
        Decode audio from bytes.
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            Audio as numpy array
        """
        # Try to decode as WAV first
        try:
            with io.BytesIO(audio_bytes) as audio_io:
                with wave.open(audio_io, 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    return audio_data
        except:
            # Try other formats using soundfile
            import soundfile as sf
            with io.BytesIO(audio_bytes) as audio_io:
                audio_data, sample_rate = sf.read(audio_io)
                
                # Resample if needed
                if sample_rate != self.sample_rate:
                    audio_data = await self.audio_processor.resample(
                        audio_data,
                        original_rate=sample_rate,
                        target_rate=self.sample_rate
                    )
                
                return audio_data
    
    def _encode_audio(self, audio_data: np.ndarray) -> bytes:
        """
        Encode audio to bytes.
        
        Args:
            audio_data: Audio as numpy array
            
        Returns:
            Encoded audio bytes
        """
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32768).astype(np.int16)
        
        # Create WAV file in memory
        with io.BytesIO() as audio_io:
            with wave.open(audio_io, 'wb') as wav:
                wav.setnchannels(1)  # Mono
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(self.sample_rate)
                wav.writeframes(audio_int16.tobytes())
            
            return audio_io.getvalue()
    
    async def _download_audio(self, url: str) -> np.ndarray:
        """
        Download audio from URL.
        
        Args:
            url: Audio URL
            
        Returns:
            Audio as numpy array
        """
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    audio_bytes = await response.read()
                    return await self._decode_audio_bytes(audio_bytes)
                else:
                    raise ValueError(f"Failed to download audio: HTTP {response.status}")
    
    def _format_result(self, result: AudioResult) -> Dict[str, Any]:
        """
        Format audio result for response.
        
        Args:
            result: AudioResult object
            
        Returns:
            Formatted dictionary
        """
        output = {
            "task": result.task.value,
            "confidence": result.confidence,
            "processing_time_ms": result.processing_time_ms,
            "sample_rate": result.sample_rate,
            "duration_seconds": result.duration_seconds
        }
        
        if result.text:
            output["text"] = result.text
        
        if result.audio_data:
            # Encode audio as base64 for transmission
            output["audio"] = base64.b64encode(result.audio_data).decode('utf-8')
            output["audio_format"] = "wav"
        
        if result.classifications:
            output["classifications"] = result.classifications
        
        if result.speaker_id:
            output["speaker_id"] = result.speaker_id
        
        if result.emotions:
            output["emotions"] = result.emotions
        
        if result.metadata:
            output["metadata"] = result.metadata
        
        return output
    
    def get_capabilities(self) -> List[ModuleCapability]:
        """
        Get module capabilities.
        
        Returns:
            List of supported capabilities
        """
        return [
            ModuleCapability.AUDIO_PROCESSING,
            ModuleCapability.SPEECH_TO_TEXT,
            ModuleCapability.TEXT_TO_SPEECH
        ]
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up Audio Module resources")
        
        # Clear cache
        self.audio_cache.clear()
        
        # Cleanup processors
        if hasattr(self, 'speech_to_text'):
            await self.speech_to_text.cleanup()
        if hasattr(self, 'text_to_speech'):
            await self.text_to_speech.cleanup()
        
        self._initialized = False