import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from app.config import DetectionConfig
from app.edgetpu import create_interpreter

logger = logging.getLogger(__name__)

MODELS_DIR = "/models"
BIRD_CLASS_ID = 16  # COCO class ID for "bird"


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Detection:
    bbox: BBox
    confidence: float


class BirdDetector:
    def __init__(self, config: DetectionConfig, db=None):
        cpu_path = os.path.join(MODELS_DIR, config.model)
        edgetpu_path = os.path.join(MODELS_DIR, config.edgetpu_model)

        self._interpreter, self._using_edgetpu = create_interpreter(cpu_path, edgetpu_path)
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._input_size = config.input_size
        self._confidence_threshold = config.bird_confidence
        self._db = db
        self._config = config

        input_shape = self._input_details[0]["shape"]
        logger.info(
            "BirdDetector ready (Edge TPU: %s, input: %s)",
            self._using_edgetpu, input_shape,
        )

    @property
    def using_edgetpu(self) -> bool:
        return self._using_edgetpu

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]

        # Get dynamic confidence threshold from database
        confidence_threshold = self._confidence_threshold
        detection_zones = []
        if self._db:
            confidence_threshold = self._db.get_setting("bird_confidence", self._confidence_threshold)
            detection_zones = self._db.get_setting("detection_zones", [])

        input_image = cv2.resize(frame, (self._input_size, self._input_size))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_image, axis=0).astype(np.uint8)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()

        # EfficientDet-Lite output format:
        #   [0] boxes: [1, N, 4] as [ymin, xmin, ymax, xmax] normalized 0-1
        #   [1] classes: [1, N]
        #   [2] scores: [1, N]
        #   [3] count: [1]
        boxes = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        classes = self._interpreter.get_tensor(self._output_details[1]["index"])[0]
        scores = self._interpreter.get_tensor(self._output_details[2]["index"])[0]
        count = int(self._interpreter.get_tensor(self._output_details[3]["index"])[0])

        detections = []
        for i in range(count):
            class_id = int(classes[i])
            score = float(scores[i])

            if class_id != BIRD_CLASS_ID:
                continue
            if score < confidence_threshold:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            bbox = BBox(
                x1=max(0, int(xmin * w)),
                y1=max(0, int(ymin * h)),
                x2=min(w, int(xmax * w)),
                y2=min(h, int(ymax * h)),
            )

            if (bbox.x2 - bbox.x1) < 10 or (bbox.y2 - bbox.y1) < 10:
                continue

            # Check if detection is within zone (if zone is defined)
            if detection_zones and not self._is_in_zone(bbox, detection_zones, w, h):
                continue

            logger.debug("Bird detected: confidence=%.3f, bbox=(%d,%d,%d,%d)",
                        score, bbox.x1, bbox.y1, bbox.x2, bbox.y2)
            detections.append(Detection(bbox=bbox, confidence=score))

        return detections

    def _is_in_zone(self, bbox: BBox, zone_points: List[dict], img_width: int, img_height: int) -> bool:
        """Check if the center of bbox is inside the detection zone polygon."""
        if not zone_points or len(zone_points) < 3:
            return True

        # Calculate center of bbox in normalized coordinates
        center_x = ((bbox.x1 + bbox.x2) / 2) / img_width
        center_y = ((bbox.y1 + bbox.y2) / 2) / img_height

        # Point-in-polygon test using ray casting algorithm
        inside = False
        n = len(zone_points)
        p1x, p1y = zone_points[0]['x'], zone_points[0]['y']

        for i in range(1, n + 1):
            p2x, p2y = zone_points[i % n]['x'], zone_points[i % n]['y']
            if center_y > min(p1y, p2y):
                if center_y <= max(p1y, p2y):
                    if center_x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (center_y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or center_x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
