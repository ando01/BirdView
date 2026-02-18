import logging
import os
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np

from app.classifier import BirdClassifier
from app.config import AppConfig
from app.db import Database
from app.frigate import FrigateClient
from app.storage import ImageStorage

logger = logging.getLogger(__name__)


class FrigatePipeline:
    """Processes Frigate bird events: classify, store, and notify."""

    def __init__(
        self,
        config: AppConfig,
        frigate_client: FrigateClient,
        classifier: BirdClassifier,
        db: Database,
        storage: ImageStorage,
        on_event_complete: Optional[Callable] = None,
    ):
        self._config = config
        self._frigate = frigate_client
        self._classifier = classifier
        self._db = db
        self._storage = storage
        self._on_event_complete = on_event_complete
        self._semaphore = threading.Semaphore(4)
        self._lock = threading.Lock()
        self._events_today = 0
        self._last_detection_info = None
        self._current_date = datetime.now().date()

    @property
    def events_today(self) -> int:
        with self._lock:
            return self._events_today

    @property
    def last_detection_info(self) -> Optional[dict]:
        with self._lock:
            return self._last_detection_info

    def process_event(self, frigate_event: dict):
        """Called by FrigateConsumer. Spawns a background thread per event."""
        t = threading.Thread(
            target=self._do_process,
            args=(frigate_event,),
            daemon=True,
            name=f"FrigatePipeline-{frigate_event['event_id'][:8]}",
        )
        t.start()

    def _do_process(self, event: dict):
        event_id = event["event_id"]
        self._semaphore.acquire()
        try:
            self._process_event_inner(event)
        except Exception:
            logger.exception("Error processing Frigate event %s", event_id[:8])
        finally:
            self._semaphore.release()

    def _process_event_inner(self, event: dict):
        event_id = event["event_id"]
        start_time = event.get("start_time")
        end_time = event.get("end_time")

        detection_time = datetime.now()
        if start_time:
            try:
                detection_time = datetime.fromtimestamp(start_time)
            except (TypeError, ValueError, OSError):
                pass

        duration = 0.0
        if start_time and end_time:
            try:
                duration = float(end_time) - float(start_time)
            except (TypeError, ValueError):
                pass

        # Download snapshot
        jpeg_bytes = self._frigate.get_snapshot_bytes(event_id)
        if jpeg_bytes is None:
            logger.warning("No snapshot for event %s, skipping", event_id[:8])
            return

        # Decode JPEG to numpy array for classification
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.warning("Failed to decode snapshot for event %s", event_id[:8])
            return

        # Extract bird crop using Frigate box [x1, y1, x2, y2] (pixel coords)
        box = event.get("box", [])
        crop = frame
        bbox = None
        if len(box) == 4:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                bbox = (x1, y1, x2, y2)

        # Classify
        classification = self._classifier.classify(crop)
        if classification is None:
            logger.debug("Event %s: below classification threshold, skipping", event_id[:8])
            return

        # Save snapshot bytes (raw JPEG from Frigate)
        snapshot_path = self._storage.save_snapshot_bytes(
            event_id, jpeg_bytes, detection_time
        )

        # Save thumbnail
        thumbnail_path = None
        if bbox is not None:
            thumbnail_path = self._storage.save_thumbnail(
                event_id, frame, bbox, detection_time
            )
        else:
            thumbnail_path = self._storage.save_thumbnail(
                event_id, frame, (0, 0, frame.shape[1], frame.shape[0]), detection_time
            )

        # Insert into database
        self._db.insert_detection(
            event_id=event_id,
            detection_time=detection_time,
            duration_seconds=duration,
            score=classification.score,
            scientific_name=classification.scientific_name,
            common_name=classification.common_name,
            detection_confidence=event.get("frigate_score", 0.0),
            snapshot_path=snapshot_path,
            thumbnail_path=thumbnail_path,
        )

        # Update counters
        with self._lock:
            today = datetime.now().date()
            if today != self._current_date:
                self._events_today = 0
                self._current_date = today
            self._events_today += 1
            self._last_detection_info = {
                "species": classification.common_name,
                "score": classification.score,
                "time": detection_time.strftime("%H:%M:%S"),
            }

        logger.info(
            "Event %s: %s (score=%.2f, duration=%.1fs, camera=%s)",
            event_id[:8],
            classification.common_name,
            classification.score,
            duration,
            event.get("camera", "?"),
        )

        # Async clip download
        if self._config.frigate.download_clips and event.get("has_clip"):
            delay = self._config.frigate.clip_download_delay

            def _download_clip():
                time.sleep(delay)
                tmp_path = os.path.join("/tmp", f"{event_id}_clip.mp4")
                if self._frigate.download_clip(event_id, tmp_path):
                    try:
                        rel_path = self._storage.save_clip(event_id, tmp_path, detection_time)
                        self._db.update_clip_path(event_id, rel_path)
                        logger.info("Clip saved for event %s", event_id[:8])
                    except Exception:
                        logger.exception("Error saving clip for %s", event_id[:8])

            clip_thread = threading.Thread(target=_download_clip, daemon=True)
            clip_thread.start()

        # Notify callback
        if self._on_event_complete:
            try:
                self._on_event_complete(event_id, classification, detection_time, duration,
                                        event.get("frigate_score", 0.0), snapshot_path)
            except Exception:
                logger.exception("Error in on_event_complete for %s", event_id[:8])
