import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

try:
    from core.npu_runtime import NPURuntime
except ImportError:
    NPURuntime = None

logger = logging.getLogger(__name__)

__all__ = ['YOLOVisionGuard', 'Detection', 'ScreenScanResult', 'PrivacyAlert']


@dataclass
class Detection:
    bbox: List[float]  # [x_min, y_min, x_max, y_max]
    confidence: float
    class_name: str
    class_id: int


@dataclass
class ScreenScanResult:
    detections: List[Detection]
    privacy_risk_level: str
    sensitive_regions: List[List[float]]


@dataclass
class PrivacyAlert:
    alert_level: str
    detected_items: List[str]
    recommendations: List[str]


class YOLOVisionGuard:
    """Screen Guard & Vision using YOLOv11 Nano."""
    
    COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    
    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.model = None
        
        if not self.demo_mode and NPURuntime:
            try:
                self.runtime = NPURuntime()
                self.model = self.runtime.load_model("models/yolov11_nano.onnx")
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}. Using demo mode.")
                self.demo_mode = True
        else:
            self.demo_mode = True

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # Dummy preprocess: resize 640x640, norm, CHW
        return np.zeros((1, 3, 640, 640), dtype=np.float32)

    def nms(self, boxes, scores, iou_threshold=0.45):
        # Dummy NMS
        return []

    def detect(self, image: np.ndarray) -> List[Detection]:
        if self.demo_mode:
            # Synthetic detections
            return [
                Detection(bbox=[10.0, 20.0, 100.0, 200.0], confidence=0.85, class_name="person", class_id=0),
                Detection(bbox=[300.0, 150.0, 450.0, 300.0], confidence=0.75, class_name="cell phone", class_id=67)
            ]
        
        # Inference logic here
        return []

    def scan_screen(self) -> ScreenScanResult:
        # Mock screen capture
        dummy_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = self.detect(dummy_screen)
        
        risk = "safe"
        sensitive = []
        for d in detections:
            if d.class_name in ["cell phone", "person"]:
                risk = "caution"
                sensitive.append(d.bbox)
                
        return ScreenScanResult(
            detections=detections,
            privacy_risk_level=risk,
            sensitive_regions=sensitive
        )

    def check_privacy(self, image: np.ndarray) -> PrivacyAlert:
        detections = self.detect(image)
        detected_items = [d.class_name for d in detections]
        
        alert = "safe"
        recs = []
        
        if "person" in detected_items:
            alert = "caution"
            recs.append("Someone might be looking at your screen.")
        if "cell phone" in detected_items:
            alert = "danger"
            recs.append("Camera/phone detected! Potential recording in progress.")
            
        return PrivacyAlert(
            alert_level=alert,
            detected_items=list(set(detected_items)),
            recommendations=recs
        )
