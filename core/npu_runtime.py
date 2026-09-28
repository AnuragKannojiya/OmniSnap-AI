import time
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    logger.warning("onnxruntime not installed. Running in mock/stub mode.")
    ort = None
    HAS_ORT = False

@dataclass
class NPUTelemetry:
    """Tracks metrics for NPU and overall AI engine utilization."""
    active_models: List[str] = field(default_factory=list)
    inference_count: int = 0
    avg_latency_ms: Dict[str, float] = field(default_factory=dict)
    _model_inference_counts: Dict[str, int] = field(default_factory=dict)
    _model_total_latency: Dict[str, float] = field(default_factory=dict)
    estimated_power_draw_w: float = 0.0
    npu_utilization_pct: float = 0.0
    start_time: float = field(default_factory=time.time)

    def record_inference(self, model_name: str, latency_ms: float, provider: str):
        """Records an inference execution and updates telemetry."""
        self.inference_count += 1
        
        # Update model-specific latency averages
        self._model_inference_counts[model_name] = self._model_inference_counts.get(model_name, 0) + 1
        self._model_total_latency[model_name] = self._model_total_latency.get(model_name, 0.0) + latency_ms
        self.avg_latency_ms[model_name] = self._model_total_latency[model_name] / self._model_inference_counts[model_name]

        # Estimate power draw based on the provider used
        if provider == 'QnnExecutionProvider':
            self.estimated_power_draw_w = 3.8
            self.npu_utilization_pct = min(100.0, self.npu_utilization_pct + 2.5)  # simplistic util model
        elif provider == 'DmlExecutionProvider':
            self.estimated_power_draw_w = 15.0
            self.npu_utilization_pct = 0.0
        else:
            self.estimated_power_draw_w = 35.0
            self.npu_utilization_pct = 0.0

    def decay_utilization(self):
        """Periodically decay the NPU utilization metric."""
        self.npu_utilization_pct = max(0.0, self.npu_utilization_pct - 5.0)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

class NPURuntime:
    """Central runtime abstraction managing ONNX Runtime sessions with Snapdragon Hexagon NPU."""

    def __init__(self):
        self.telemetry = NPUTelemetry()
        self._lock = threading.Lock()
        self._active_sessions = {}
        
        self.available_providers = []
        if HAS_ORT:
            self.available_providers = ort.get_available_providers()
            logger.info(f"Available ORT execution providers: {self.available_providers}")

    def create_session(self, model_name: str, model_path: str) -> Any:
        """Creates an ONNX Runtime InferenceSession with the proper provider chain."""
        with self._lock:
            if model_name in self._active_sessions:
                return self._active_sessions[model_name]

            if not HAS_ORT:
                logger.info(f"Mocking session creation for {model_name} at {model_path}")
                self.telemetry.active_models.append(model_name)
                session = "mock_session"
                self._active_sessions[model_name] = session
                return session

            # Configure QNN HTP backend options
            qnn_options = {
                'backend_path': 'QnnHtp.dll',
                'htp_performance_mode': 'burst',
                'htp_graph_finalization_optimization_mode': '3',
                'enable_htp_fp16_precision': '1',
            }

            providers = []
            provider_options = []

            if 'QnnExecutionProvider' in self.available_providers:
                providers.append('QnnExecutionProvider')
                provider_options.append(qnn_options)
            
            if 'DmlExecutionProvider' in self.available_providers:
                providers.append('DmlExecutionProvider')
                provider_options.append({})
                
            providers.append('CPUExecutionProvider')
            provider_options.append({})

            try:
                session = ort.InferenceSession(
                    model_path,
                    providers=providers,
                    provider_options=provider_options
                )
                self.telemetry.active_models.append(model_name)
                self._active_sessions[model_name] = session
                logger.info(f"Successfully created session for {model_name}")
                return session
            except Exception as e:
                logger.error(f"Failed to create session for {model_name}: {e}")
                raise

    def get_device_info(self) -> Dict[str, Any]:
        """Returns information about the available hardware execution providers."""
        return {
            "has_onnxruntime": HAS_ORT,
            "available_providers": self.available_providers,
            "has_qnn": 'QnnExecutionProvider' in self.available_providers,
            "has_dml": 'DmlExecutionProvider' in self.available_providers
        }

    def get_npu_telemetry(self) -> Dict[str, Any]:
        """Retrieves the current telemetry metrics."""
        # Optional: Decay utilization here for simplicity if a background thread isn't polling
        self.telemetry.decay_utilization()
        
        return {
            "active_models": self.telemetry.active_models,
            "inference_count": self.telemetry.inference_count,
            "avg_latency_ms": self.telemetry.avg_latency_ms,
            "estimated_power_draw_w": self.telemetry.estimated_power_draw_w,
            "npu_utilization_pct": self.telemetry.npu_utilization_pct,
            "uptime_seconds": self.telemetry.uptime_seconds
        }

    def run_inference(self, model_name: str, inputs: Dict[str, Any]) -> Any:
        """Executes a model inference, tracking telemetry."""
        if model_name not in self._active_sessions:
            raise ValueError(f"Model {model_name} not loaded. Create session first.")
        
        session = self._active_sessions[model_name]
        start_time = time.time()
        
        if not HAS_ORT:
            # Simulate latency in mock mode
            time.sleep(0.05)
            latency_ms = (time.time() - start_time) * 1000
            self.telemetry.record_inference(model_name, latency_ms, 'MockExecutionProvider')
            return {"mock_output": "synthetic_data"}
        
        # Get actual active provider for telemetry
        active_provider = session.get_providers()[0]
        
        try:
            output = session.run(None, inputs)
            latency_ms = (time.time() - start_time) * 1000
            self.telemetry.record_inference(model_name, latency_ms, active_provider)
            return output
        except Exception as e:
            logger.error(f"Inference failed for {model_name}: {e}")
            raise
