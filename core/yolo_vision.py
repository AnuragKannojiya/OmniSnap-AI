import logging
import time
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ['YOLOVisionGuard', 'Detection', 'ScreenScanResult', 'PrivacyAlert']

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    logger.warning("ultralytics not available. YOLO will use demo mode.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    class_id: int

@dataclass
class ScreenScanResult:
    detections: List[Detection]
    privacy_risk_level: str  # safe, caution, danger
    sensitive_regions: List[List[float]]
    num_objects: int
    latency_ms: float

@dataclass
class PrivacyAlert:
    alert_level: str  # safe, caution, danger
    detected_items: List[str]
    recommendations: List[str]
    person_count: int = 0
    device_count: int = 0


# Privacy-sensitive COCO classes
PRIVACY_DANGER_CLASSES = {'cell phone', 'laptop', 'tv', 'monitor'}
PRIVACY_CAUTION_CLASSES = {'person', 'book', 'keyboard', 'mouse', 'remote'}


class YOLOVisionGuard:
    """Screen Guard & Vision using YOLOv11 Nano.
    
    Uses ultralytics YOLO for real object detection.
    Auto-downloads yolo11n.pt (~6MB) on first use.
    """
    
    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = 0.35, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.model = None
        self.confidence = confidence
        self._inference_count = 0
        self._total_latency = 0.0
        
        if demo_mode or not HAS_ULTRALYTICS:
            self.demo_mode = True
            logger.info("YOLOVisionGuard initialized in demo mode")
            return
        
        try:
            logger.info(f"Loading YOLO model '{model_path}'...")
            self.model = YOLO(model_path)
            logger.info(f"✅ YOLO model loaded ({model_path})")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.demo_mode = True
    
    def detect(self, image: Union[np.ndarray, str, 'Image.Image'], confidence: float = None) -> List[Detection]:
        """Detect objects in an image.
        
        Args:
            image: numpy array (BGR/RGB), file path, or PIL Image
            confidence: optional confidence threshold override
        """
        if self.demo_mode:
            return self._demo_detect()
        
        try:
            conf = confidence or self.confidence
            results = self.model(image, conf=conf, verbose=False)
            
            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    detections.append(Detection(
                        bbox=box.xyxy[0].tolist(),
                        confidence=float(box.conf[0]),
                        class_name=self.model.names.get(cls_id, f"class_{cls_id}"),
                        class_id=cls_id,
                    ))
            
            self._inference_count += 1
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def detect_from_bytes(self, image_bytes: bytes) -> List[Detection]:
        """Detect objects from image bytes (e.g., from file upload)."""
        if self.demo_mode:
            return self._demo_detect()
        
        try:
            if HAS_PIL:
                import io
                img = Image.open(io.BytesIO(image_bytes))
                return self.detect(img)
            else:
                # Decode with numpy/cv2
                import cv2
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return []
                return self.detect(img)
        except Exception as e:
            logger.error(f"Bytes detection error: {e}")
            return []
    
    def scan_screen(self, image: Union[np.ndarray, str, 'Image.Image'] = None) -> ScreenScanResult:
        """Scan a screen image and analyze for privacy concerns."""
        start = time.time()
        
        if image is None:
            # If no image provided, use a dummy for demo
            detections = self._demo_detect()
        else:
            detections = self.detect(image)
        
        latency = (time.time() - start) * 1000
        
        # Determine privacy risk
        risk = "safe"
        sensitive = []
        for d in detections:
            if d.class_name in PRIVACY_DANGER_CLASSES:
                risk = "danger"
                sensitive.append(d.bbox)
            elif d.class_name in PRIVACY_CAUTION_CLASSES and risk != "danger":
                risk = "caution"
                sensitive.append(d.bbox)
        
        return ScreenScanResult(
            detections=detections,
            privacy_risk_level=risk,
            sensitive_regions=sensitive,
            num_objects=len(detections),
            latency_ms=round(latency, 1),
        )
    
    def check_privacy(self, image: Union[np.ndarray, str, 'Image.Image'] = None) -> PrivacyAlert:
        """Check image for privacy-sensitive content."""
        if image is not None:
            detections = self.detect(image)
        else:
            detections = self._demo_detect()
        
        detected_items = list(set(d.class_name for d in detections))
        person_count = sum(1 for d in detections if d.class_name == 'person')
        device_count = sum(1 for d in detections if d.class_name in PRIVACY_DANGER_CLASSES)
        
        alert = "safe"
        recs = []
        
        if device_count > 0:
            alert = "danger"
            recs.append("📱 Camera/recording device detected! Verify no unauthorized recording is in progress.")
        if person_count > 1:
            alert = "danger" if alert != "danger" else alert
            recs.append(f"👥 {person_count} people detected — potential shoulder surfing risk.")
        elif person_count == 1:
            if alert == "safe":
                alert = "caution"
            recs.append("👤 One person detected — verify it's only the authorized user.")
        
        if alert == "safe":
            recs.append("✅ No privacy threats detected. Screen sharing is safe.")
        
        return PrivacyAlert(
            alert_level=alert,
            detected_items=detected_items,
            recommendations=recs,
            person_count=person_count,
            device_count=device_count,
        )
    
    def _demo_detect(self) -> List[Detection]:
        """Return demo detections."""
        return [
            Detection(bbox=[45.0, 120.0, 280.0, 450.0], confidence=0.92, class_name="person", class_id=0),
            Detection(bbox=[320.0, 200.0, 480.0, 360.0], confidence=0.85, class_name="laptop", class_id=63),
            Detection(bbox=[500.0, 180.0, 620.0, 290.0], confidence=0.78, class_name="cell phone", class_id=67),
        ]
    
    @property
    def avg_latency_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return self._total_latency / self._inference_count
