import logging
import threading
import time
from datetime import datetime, timedelta
from typing import List, Optional

from app.camera import CameraStream
from app.classifier import BirdClassifier
from app.clip import ClipEncoder
from app.config import AppConfig
from app.db import Database
from app.detector import BirdDetector
from app.motion import MotionDetector
from app.storage import ImageStorage
from app.tracker import EventTracker, TrackedBird

logger = logging.getLogger(__name__)


class DetectionPipeline(threading.Thread):
    """Main detection loop: camera -> motion -> detect -> classify -> track -> store."""

    def __init__(
        self,
        config: AppConfig,
        camera: CameraStream,
        motion: Optional[MotionDetector],
        detector: BirdDetector,
        classifier: BirdClassifier,
        tracker: EventTracker,
        db: Database,
        storage: ImageStorage,
        clip_encoder: Optional[ClipEncoder],
        on_event_complete=None,
    ):
        super().__init__(daemon=True, name="DetectionPipeline")
        self._config = config
        self._camera = camera
        self._motion = motion
        self._detector = detector
        self._classifier = classifier
        self._tracker = tracker
        self._db = db
        self._storage = storage
        self._clip_encoder = clip_encoder
        self._on_event_complete = on_event_complete
        self._running = True
        self._frame_interval = 1.0 / config.camera.detection_fps
        self._events_today = 0
        self._last_detection_info = None

    @property
    def events_today(self) -> int:
        return self._events_today

    @property
    def last_detection_info(self) -> Optional[dict]:
        return self._last_detection_info

    @property
    def active_birds(self) -> int:
        return self._tracker.active_count

    @property
    def detecting(self) -> bool:
        return self._tracker.active_count > 0

    def run(self):
        last_process_time = 0.0
        current_date = datetime.now().date()

        logger.info(
            "Detection pipeline started (%.1f FPS, motion=%s, clips=%s)",
            self._config.camera.detection_fps,
            "on" if self._motion else "off",
            "on" if self._clip_encoder and self._config.clips.enabled else "off",
        )

        while self._running:
            now = time.time()
            if now - last_process_time < self._frame_interval:
                time.sleep(0.01)
                continue

            frame = self._camera.get_frame()
            if frame is None:
                time.sleep(0.5)
                continue

            last_process_time = now
            timestamp = datetime.now()

            # Reset daily counter
            if timestamp.date() != current_date:
                self._events_today = 0
                current_date = timestamp.date()

            # Step 1: Motion pre-filter
            if self._motion and not self._motion.has_motion(frame):
                completed = self._tracker.update([], [], frame, timestamp)
                self._finalize_events(completed)
                continue

            # Step 2: Object detection
            try:
                detections = self._detector.detect(frame)
            except Exception:
                logger.exception("Error in object detection")
                continue

            if not detections:
                completed = self._tracker.update([], [], frame, timestamp)
                self._finalize_events(completed)
                continue

            logger.debug("Found %d bird detection(s), classifying...", len(detections))

            # Step 3: Classify each detected bird
            classifications = []
            for det in detections:
                try:
                    crop = frame[det.bbox.y1 : det.bbox.y2, det.bbox.x1 : det.bbox.x2]
                    if crop.size == 0:
                        classifications.append(None)
                        continue
                    cls = self._classifier.classify(crop)
                    classifications.append(cls)
                except Exception:
                    logger.exception("Error classifying bird")
                    classifications.append(None)

            # Step 4: Update tracker
            completed = self._tracker.update(
                detections, classifications, frame, timestamp
            )

            # Step 5: Finalize completed events
            self._finalize_events(completed)

    def _finalize_events(self, completed_events: List[TrackedBird]):
        for event in completed_events:
            if event.best_classification is None:
                continue

            try:
                self._save_event(event)
                self._events_today += 1
            except Exception:
                logger.exception("Error saving event %s", event.event_id[:8])

    def _save_event(self, event: TrackedBird):
        cls = event.best_classification
        duration = (event.last_seen - event.first_seen).total_seconds()

        # Save snapshot and thumbnail
        snapshot_path = None
        thumbnail_path = None
        if event.best_snapshot is not None:
            snapshot_path = self._storage.save_snapshot(
                event.event_id, event.best_snapshot, event.first_seen
            )
            thumbnail_path = self._storage.save_thumbnail(
                event.event_id,
                event.best_snapshot,
                (event.bbox.x1, event.bbox.y1, event.bbox.x2, event.bbox.y2),
                event.first_seen,
            )

        # Insert into database (clip_path added later async)
        self._db.insert_detection(
            event_id=event.event_id,
            detection_time=event.first_seen,
            duration_seconds=duration,
            score=cls.score,
            scientific_name=cls.scientific_name,
            common_name=cls.common_name,
            detection_confidence=event.best_detection_conf,
            snapshot_path=snapshot_path,
            thumbnail_path=thumbnail_path,
        )

        self._last_detection_info = {
            "species": cls.common_name,
            "score": cls.score,
            "time": event.first_seen.strftime("%H:%M:%S"),
        }

        logger.info(
            "Event %s: %s (score=%.2f, duration=%.1fs)",
            event.event_id[:8],
            cls.common_name,
            cls.score,
            duration,
        )

        # Encode video clip asynchronously
        if self._clip_encoder and self._config.clips.enabled:
            pre = self._config.clips.pre_capture
            post = self._config.clips.post_capture
            start = event.first_seen - timedelta(seconds=pre)
            end = event.last_seen + timedelta(seconds=post)

            # Cap duration
            max_dur = self._config.clips.max_duration
            if (end - start).total_seconds() > max_dur:
                end = start + timedelta(seconds=max_dur)

            clip_frames = self._camera.get_clip_frames(start, end)
            if clip_frames:

                def _on_clip_encoded(eid, tmp_path):
                    try:
                        rel_path = self._storage.save_clip(
                            eid, tmp_path, event.first_seen
                        )
                        self._db.update_clip_path(eid, rel_path)
                        logger.info(
                            "Clip saved for event %s (%d frames)",
                            eid[:8],
                            len(clip_frames),
                        )
                    except Exception:
                        logger.exception("Error saving clip for %s", eid[:8])

                self._clip_encoder.encode_clip_async(
                    clip_frames, event.event_id, _on_clip_encoded
                )

        # Notify MQTT callback
        if self._on_event_complete:
            self._on_event_complete(event, snapshot_path)

    def stop(self):
        self._running = False
