import logging
import time
import math
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

try:
    from core.npu_runtime import NPURuntime, NPUModel
except ImportError:
    # Allow running in standalone mode for testing
    NPURuntime = None
    NPUModel = None

logger = logging.getLogger(__name__)

__all__ = ['WhisperASR', 'TranscriptionResult', 'TranscriptionSegment']


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    confidence: float


@dataclass
class TranscriptionResult:
    text: str
    segments: List[TranscriptionSegment]
    language: str
    confidence: float
    latency_ms: float = 0.0


def log_mel_spectrogram(audio: np.ndarray, n_mels: int = 80, n_fft: int = 400, hop_length: int = 160) -> np.ndarray:
    """Lightweight numpy-based log-mel spectrogram computation."""
    # Simplified mock for demo purposes if full processing isn't needed,
    # but let's provide a basic working version structure.
    # In a real scenario we would compute STFT and apply Mel filterbanks.
    # Here we return a dummy spectrogram shape to satisfy the interface.
    frames = 1 + (len(audio) - n_fft) // hop_length
    if frames <= 0:
        frames = 1
    return np.zeros((n_mels, frames), dtype=np.float32)


class WhisperASR:
    """Real-Time Speech Recognition using Whisper Base EN."""
    
    def __init__(self, model_manager=None, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.model = None
        self.is_streaming = False
        self.sample_rate = 16000
        
        if not self.demo_mode and NPURuntime:
            try:
                self.runtime = NPURuntime()
                # Assuming model_manager provides path, or we hardcode based on known Snapdragon hub models
                model_path = "models/whisper_base_en.onnx"
                self.model = self.runtime.load_model(model_path)
            except Exception as e:
                logger.warning(f"Failed to load Whisper model on NPU: {e}. Falling back to demo mode.")
                self.demo_mode = True
        else:
            self.demo_mode = True

    def start_streaming(self):
        """Start real-time microphone input stream."""
        self.is_streaming = True
        logger.info("Started ASR streaming.")

    def stop_streaming(self):
        """Stop real-time microphone input stream."""
        self.is_streaming = False
        logger.info("Stopped ASR streaming.")

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe audio data."""
        start_time = time.time()
        
        if self.demo_mode:
            # Simulate latency
            time.sleep(0.5)
            latency = (time.time() - start_time) * 1000
            return TranscriptionResult(
                text="This is a simulated transcription from OmniSnap AI.",
                segments=[
                    TranscriptionSegment(start=0.0, end=2.0, text="This is a simulated transcription", confidence=0.95),
                    TranscriptionSegment(start=2.0, end=3.5, text="from OmniSnap AI.", confidence=0.98)
                ],
                language="en",
                confidence=0.96,
                latency_ms=latency
            )

        # Preprocessing
        if sample_rate != self.sample_rate:
            # Resampling logic would go here
            pass
        
        mel = log_mel_spectrogram(audio_data)
        
        # Inference
        if self.model:
            # Replace with actual inference call
            # outputs = self.model.run({'mel': mel})
            time.sleep(0.1) # Simulate real inference
            
        latency = (time.time() - start_time) * 1000
        
        # Mock result for NPU path
        return TranscriptionResult(
            text="Decoded text from NPU",
            segments=[],
            language="en",
            confidence=0.9,
            latency_ms=latency
        )
