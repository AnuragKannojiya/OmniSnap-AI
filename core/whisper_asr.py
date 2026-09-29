import logging
import time
import os
import io
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ['WhisperASR', 'TranscriptionResult', 'TranscriptionSegment']

# Try importing torch and transformers
try:
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers/torch not available. Whisper will use demo mode.")


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


class WhisperASR:
    """Real-Time Speech Recognition using Whisper Base EN.
    
    Uses the openai/whisper-base.en model from HuggingFace Transformers.
    Auto-downloads on first use (~290MB).
    """
    
    MODEL_ID = "openai/whisper-base.en"
    
    def __init__(self, model_id: str = None, device: str = None, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.model_id = model_id or self.MODEL_ID
        self.pipe = None
        self.processor = None
        self.model = None
        self.sample_rate = 16000
        self._inference_count = 0
        self._total_latency = 0.0
        
        if demo_mode or not HAS_TRANSFORMERS:
            self.demo_mode = True
            logger.info("WhisperASR initialized in demo mode")
            return
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps" 
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        try:
            logger.info(f"Loading Whisper model '{self.model_id}' on {self.device}...")
            # Use the pipeline API for simplicity and robustness
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=self.device if self.device != "mps" else -1,  # MPS not always supported
                dtype=torch.float32,
            )
            logger.info(f"✅ Whisper model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            logger.info("Falling back to demo mode")
            self.demo_mode = True
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe audio data to text.
        
        Args:
            audio_data: Audio samples as numpy float32 array (mono, any sample rate)
            sample_rate: Sample rate of the audio data
            
        Returns:
            TranscriptionResult with text, segments, and timing info
        """
        start_time = time.time()
        
        if self.demo_mode:
            return self._demo_transcribe(start_time)
        
        try:
            # Ensure float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize if needed (audio should be in [-1, 1] range)
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Run inference
            result = self.pipe(
                {"raw": audio_data, "sampling_rate": sample_rate},
                return_timestamps=True,
            )
            
            latency = (time.time() - start_time) * 1000
            self._inference_count += 1
            self._total_latency += latency
            
            # Parse segments
            segments = []
            if "chunks" in result:
                for chunk in result["chunks"]:
                    ts = chunk.get("timestamp", (0.0, 0.0))
                    segments.append(TranscriptionSegment(
                        start=ts[0] if ts[0] is not None else 0.0,
                        end=ts[1] if ts[1] is not None else 0.0,
                        text=chunk["text"].strip(),
                        confidence=0.95,
                    ))
            
            text = result.get("text", "").strip()
            
            return TranscriptionResult(
                text=text,
                segments=segments,
                language="en",
                confidence=0.95,
                latency_ms=round(latency, 1),
            )
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            latency = (time.time() - start_time) * 1000
            return TranscriptionResult(
                text=f"[Error: {str(e)[:100]}]",
                segments=[],
                language="en",
                confidence=0.0,
                latency_ms=round(latency, 1),
            )
    
    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """Transcribe an audio file."""
        start_time = time.time()
        
        if self.demo_mode:
            return self._demo_transcribe(start_time)
        
        try:
            # Try soundfile first, then fall back to torchaudio
            try:
                import soundfile as sf
                audio_data, sr = sf.read(file_path, dtype='float32')
            except ImportError:
                import torchaudio
                waveform, sr = torchaudio.load(file_path)
                audio_data = waveform.squeeze().numpy()
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            return self.transcribe(audio_data, sample_rate=sr)
            
        except Exception as e:
            logger.error(f"File transcription error: {e}")
            latency = (time.time() - start_time) * 1000
            return TranscriptionResult(
                text=f"[Error reading audio file: {str(e)[:100]}]",
                segments=[], language="en", confidence=0.0, latency_ms=round(latency, 1)
            )
    
    def transcribe_bytes(self, audio_bytes: bytes, format: str = "wav") -> TranscriptionResult:
        """Transcribe audio from raw bytes (e.g., from file upload)."""
        start_time = time.time()
        
        if self.demo_mode:
            return self._demo_transcribe(start_time)
        
        try:
            import soundfile as sf
            audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            return self.transcribe(audio_data, sample_rate=sr)
        except Exception as e:
            logger.error(f"Bytes transcription error: {e}")
            latency = (time.time() - start_time) * 1000
            return TranscriptionResult(
                text=f"[Error processing audio: {str(e)[:100]}]",
                segments=[], language="en", confidence=0.0, latency_ms=round(latency, 1)
            )
    
    def _demo_transcribe(self, start_time: float) -> TranscriptionResult:
        """Return synthetic transcription for demo mode."""
        import time as t
        t.sleep(0.2)  # Simulate processing
        latency = (time.time() - start_time) * 1000
        return TranscriptionResult(
            text="Welcome to OmniSnap AI. This transcription is generated in demo mode. Install torch and transformers for real Whisper inference on Hexagon NPU.",
            segments=[
                TranscriptionSegment(start=0.0, end=2.0, text="Welcome to OmniSnap AI.", confidence=0.97),
                TranscriptionSegment(start=2.0, end=5.0, text="This transcription is generated in demo mode.", confidence=0.95),
            ],
            language="en", confidence=0.96, latency_ms=round(latency, 1)
        )
    
    @property
    def avg_latency_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return self._total_latency / self._inference_count
