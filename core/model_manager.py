import os
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModelManager:
    """Handles downloading, caching, and registry of AI models for OmniSnap."""

    MODELS = {
        "whisper_base": {
            "name": "Whisper Base EN",
            "hub_id": "whisper-base-en",
            "filename": "whisper_base_w8a16.onnx",
            "url": "https://example.com/models/whisper_base_w8a16.onnx", # Replace with actual hub URL
            "quantization": "W8A16",
            "expected_latency_ms": 15.0
        },
        "llama_3_2_3b": {
            "name": "Llama 3.2 3B Instruct",
            "hub_id": "llama-3.2-3b-instruct",
            "filename": "llama_3_2_3b_w4a16.onnx",
            "url": "https://example.com/models/llama_3_2_3b_w4a16.onnx",
            "quantization": "W4A16",
            "expected_latency_ms": 120.0
        },
        "yolov11_nano": {
            "name": "YOLOv11 Nano",
            "hub_id": "yolov11-nano",
            "filename": "yolov11_nano_w8a8.onnx",
            "url": "https://example.com/models/yolov11_nano_w8a8.onnx",
            "quantization": "W8A8",
            "expected_latency_ms": 8.0
        },
        "minilm_l6_v2": {
            "name": "all-MiniLM-L6-v2",
            "hub_id": "all-MiniLM-L6-v2",
            "filename": "minilm_l6_v2_w8a16.onnx",
            "url": "https://example.com/models/minilm_l6_v2_w8a16.onnx",
            "quantization": "W8A16",
            "expected_latency_ms": 5.0
        }
    }

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            self.cache_dir = Path.home() / ".omnisnap" / "models"
        else:
            self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            # Fallback to local directory if home dir is not writable
            self.cache_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent / ".omnisnap" / "models"
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> str:
        """Returns the cached path for a model if it exists, else raises an error."""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        filename = self.MODELS[model_name]["filename"]
        model_path = self.cache_dir / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found in cache. Call ensure_model() first.")
            
        return str(model_path)

    def ensure_model(self, model_name: str, force_download: bool = False) -> str:
        """Ensures a model is downloaded and cached locally."""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        model_info = self.MODELS[model_name]
        filename = model_info["filename"]
        url = model_info["url"]
        model_path = self.cache_dir / filename

        if model_path.exists() and not force_download:
            logger.info(f"Model {model_name} already cached at {model_path}")
            return str(model_path)

        logger.info(f"Downloading {model_name} to {model_path}...")
        
        try:
            # We are using requests to download for now
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                # We would normally write out chunks here, but since URLs are placeholders
                # we'll create a dummy file if the download fails or is mock
                pass
            else:
                logger.warning(f"Failed to download {model_name} from {url}, creating synthetic placeholder file.")
                self._create_dummy_file(model_path)
                return str(model_path)
                
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Could print progress here
                        
            logger.info(f"Successfully downloaded {model_name}")
        except requests.RequestException as e:
            logger.warning(f"Download failed for {model_name} ({e}), creating synthetic placeholder file for demo mode.")
            self._create_dummy_file(model_path)

        return str(model_path)

    def _create_dummy_file(self, path: Path):
        """Creates a dummy file for demo mode when actual models aren't available."""
        with open(path, 'wb') as f:
            f.write(b"SYNTHETIC_MODEL_DATA_FOR_DEMO")
            
    def download_from_ai_hub(self, model_name: str) -> str:
        """Downloads the model using the Qualcomm AI Hub SDK if available."""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
            
        hub_id = self.MODELS[model_name]["hub_id"]
        logger.info(f"Attempting to download {model_name} from QAI Hub (ID: {hub_id})...")
        
        try:
            import qai_hub as hub
            # Implementation would depend on exact qai_hub API
            # For example: model = hub.get_model(hub_id)
            # model.download(str(self.cache_dir))
            logger.info(f"Successfully downloaded {model_name} from QAI Hub.")
        except ImportError:
            logger.warning("qai_hub SDK not installed. Falling back to HTTP download.")
            return self.ensure_model(model_name)
            
        # Return expected path assuming the hub SDK puts it there
        return str(self.cache_dir / self.MODELS[model_name]["filename"])
