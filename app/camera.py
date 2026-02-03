import collections
import logging
import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config import CameraConfig

logger = logging.getLogger(__name__)


class CameraStream(threading.Thread):
    """RTSP capture thread with rolling frame buffer for clip extraction."""

    def __init__(self, config: CameraConfig):
        super().__init__(daemon=True, name="CameraStream")
        self._url = config.rtsp_url
        self._delay = config.reconnect_delay
        self._max_delay = config.max_reconnect_delay
        self._buffer_seconds = config.buffer_seconds

        self._current_delay = self._delay
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = True
        self._connected = False
        self._fps = 0.0

        # Rolling buffer: stores (timestamp, frame) tuples
        # Initial capacity; resized once we know the camera FPS
        self._buffer = collections.deque(maxlen=900)
        self._buffer_lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def fps(self) -> float:
        return self._fps

    def run(self):
        while self._running:
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                logger.warning(
                    "Cannot open RTSP stream, retrying in %.0fs", self._current_delay
                )
                self._connected = False
                time.sleep(self._current_delay)
                self._current_delay = min(self._current_delay * 2, self._max_delay)
                continue

            # Read camera FPS and resize buffer
            camera_fps = cap.get(cv2.CAP_PROP_FPS)
            if camera_fps > 0:
                self._fps = camera_fps
                max_frames = int(camera_fps * self._buffer_seconds)
                with self._buffer_lock:
                    self._buffer = collections.deque(
                        self._buffer, maxlen=max(max_frames, 300)
                    )
                logger.info(
                    "Camera FPS: %.1f, buffer: %d frames (%.0fs)",
                    camera_fps, max_frames, self._buffer_seconds,
                )

            self._current_delay = self._delay
            self._connected = True
            logger.info("Connected to RTSP stream")

            consecutive_failures = 0

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        logger.warning("Lost RTSP stream, reconnecting...")
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0
                now = datetime.now()

                with self._frame_lock:
                    self._frame = frame

                with self._buffer_lock:
                    self._buffer.append((now, frame))

            cap.release()
            self._connected = False

            if self._running:
                time.sleep(self._current_delay)
                self._current_delay = min(self._current_delay * 2, self._max_delay)

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (for detection pipeline)."""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def get_clip_frames(
        self, start_time: datetime, end_time: datetime
    ) -> List[Tuple[datetime, np.ndarray]]:
        """Extract frames from the rolling buffer between start and end times."""
        with self._buffer_lock:
            return [
                (ts, frame.copy())
                for ts, frame in self._buffer
                if start_time <= ts <= end_time
            ]

    def stop(self):
        self._running = False
