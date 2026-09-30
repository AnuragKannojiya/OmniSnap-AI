import os
import sys
import platform
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path.home() / ".omnisnap" / "models"
MODELS = {
    "whisper-base-en": {
        "url": "https://aihub.qualcomm.com/models/whisper_base_en",
        "filename": "whisper_base_en.onnx"
    },
    "yolov11": {
        "url": "https://aihub.qualcomm.com/models/yolov11_det",
        "filename": "yolov11_det.onnx"
    },
    "minilm": {
        "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
        "filename": "minilm.onnx"
    }
}

def is_windows_arm64():
    """Detect if running on Windows ARM64 (Snapdragon)."""
    return platform.system() == "Windows" and platform.machine().lower() in ["arm64", "aarch64"]

def check_qnn_availability():
    """Check if ONNX Runtime QNN Execution Provider is available."""
    try:
        import onnxruntime as ort
        return 'QNNExecutionProvider' in ort.get_available_providers()
    except ImportError:
        return False

def check_models_exist():
    """Check if required models are cached locally and provide instructions if not."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    all_exist = True
    for name, info in MODELS.items():
        model_path = MODEL_DIR / info["filename"]
        if not model_path.exists():
            all_exist = False
            logger.warning(f"Model '{name}' not found at {model_path}.")
            logger.info(f"  -> Please download from: {info['url']}")
            logger.info(f"  -> Place it as: {model_path}\n")
        else:
            logger.info(f"Model '{name}' found at {model_path}.")
    return all_exist

def get_snapdragon_config():
    """Return optimal ONNX Runtime configuration for Snapdragon NPU."""
    qnn_options = {
        'backend_path': 'QnnHtp.dll',
        'htp_performance_mode': 'burst',  
        'htp_graph_finalization_optimization_mode': '3',
        'enable_htp_fp16_precision': '1',
        'vtcm_mb': '8',
    }
    
    import onnxruntime as ort
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    return {
        "providers": [
            ('QNNExecutionProvider', qnn_options),
            ('DmlExecutionProvider', {}),
            ('CPUExecutionProvider', {})
        ],
        "session_options": session_options
    }

def main():
    logger.info("=== OmniSnap Snapdragon Deployment Readiness Report ===")
    
    # 1. Platform Check
    is_arm64 = is_windows_arm64()
    logger.info(f"Platform: {platform.system()} {platform.machine()}")
    if is_arm64:
        logger.info("[OK] Running on Windows ARM64 (Snapdragon).")
    else:
        logger.warning("[!] Not running on Windows ARM64. Optimal NPU performance requires Snapdragon.")
        
    # 2. QNN Availability
    has_qnn = check_qnn_availability()
    if has_qnn:
        logger.info("[OK] ONNX Runtime QNN Execution Provider is available.")
    else:
        logger.warning("[!] ONNX Runtime QNN Execution Provider NOT available. Will fallback to DirectML/CPU.")
        if is_arm64:
            logger.info("  -> Please install onnxruntime-qnn for optimal NPU acceleration.")
            
    # 3. Model Check
    logger.info("Checking models in ~/.omnisnap/models/...")
    models_ready = check_models_exist()
    if models_ready:
        logger.info("[OK] All required ONNX models are present.")
    else:
        logger.warning("[!] Missing models. Please follow instructions above to download them.")
        
    # 4. Configuration Review
    logger.info("Generated Snapdragon Optimal Config Preview:")
    config = get_snapdragon_config()
    logger.info(f"  Providers: {[p[0] for p in config['providers']]}")
    logger.info("  QNN Options:")
    for k, v in config['providers'][0][1].items():
        logger.info(f"    {k}: {v}")
        
    logger.info("=======================================================")

if __name__ == "__main__":
    main()
