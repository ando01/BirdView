import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np

from app.classifier import Classification
from app.config import TrackerConfig
from app.detector import BBox, Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackedBird:
    event_id: str
    bbox: BBox
    first_seen: datetime
    last_seen: datetime
    best_detection_conf: float
    best_classification: Optional[Classification] = None
    best_snapshot: Optional[np.ndarray] = None
    frames_missing: int = 0


class EventTracker:
    """IoU-based tracker that groups detections into bird visit events."""

    def __init__(self, config: TrackerConfig):
        self._max_missing = config.max_missing_frames
        self._iou_threshold = config.iou_threshold
        self._active_birds: List[TrackedBird] = []

    @property
    def active_count(self) -> int:
        return len(self._active_birds)

    def update(
        self,
        detections: List[Detection],
        classifications: List[Optional[Classification]],
        frame: np.ndarray,
        timestamp: datetime,
    ) -> List[TrackedBird]:
        """Update tracker with new detections. Returns completed events."""
        matched_ids = set()

        for det, cls in zip(detections, classifications):
            best_iou = 0.0
            best_bird = None

            for bird in self._active_birds:
                iou = self._compute_iou(det.bbox, bird.bbox)
                if iou > best_iou and iou >= self._iou_threshold:
                    best_iou = iou
                    best_bird = bird

            if best_bird is not None:
                # Update existing tracked bird
                best_bird.bbox = det.bbox
                best_bird.last_seen = timestamp
                best_bird.frames_missing = 0

                if cls and (
                    best_bird.best_classification is None
                    or cls.score > best_bird.best_classification.score
                ):
                    best_bird.best_classification = cls
                    best_bird.best_snapshot = frame.copy()
                    best_bird.best_detection_conf = det.confidence

                matched_ids.add(id(best_bird))
            else:
                # New bird event
                new_bird = TrackedBird(
                    event_id=str(uuid.uuid4()),
                    bbox=det.bbox,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    best_detection_conf=det.confidence,
                    best_classification=cls,
                    best_snapshot=frame.copy() if cls else None,
                )
                self._active_birds.append(new_bird)

        # Increment missing counter for unmatched birds
        completed = []
        remaining = []

        for bird in self._active_birds:
            if id(bird) not in matched_ids:
                bird.frames_missing += 1

            if bird.frames_missing > self._max_missing:
                if bird.best_classification is not None:
                    completed.append(bird)
                # Discard if never classified
            else:
                remaining.append(bird)

        self._active_birds = remaining
        return completed

    @staticmethod
    def _compute_iou(a: BBox, b: BBox) -> float:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0
