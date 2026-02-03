import logging
import os
import tempfile
import threading
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from app.config import ClipConfig

logger = logging.getLogger(__name__)


class ClipEncoder:
    def __init__(self, config: ClipConfig, temp_dir: str = "/tmp"):
        self._config = config
        self._temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

    def encode_clip(
        self,
        frames: List[Tuple[datetime, np.ndarray]],
        output_path: str,
    ) -> bool:
        """Encode a list of (timestamp, frame) tuples to an MP4 file.

        Returns True on success, False on failure.
        """
        if not frames:
            logger.warning("No frames to encode for clip")
            return False

        h, w = frames[0][1].shape[:2]
        fps = self._config.fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            logger.error("Failed to open VideoWriter for %s", output_path)
            return False

        try:
            for _, frame in frames:
                writer.write(frame)
            return True
        except Exception:
            logger.exception("Error encoding clip")
            return False
        finally:
            writer.release()

    def encode_clip_async(
        self,
        frames: List[Tuple[datetime, np.ndarray]],
        event_id: str,
        callback: Optional[Callable[[str, str], None]] = None,
    ):
        """Encode a clip in a background thread.

        Args:
            frames: List of (timestamp, frame) tuples.
            event_id: The event ID for naming.
            callback: Called with (event_id, temp_file_path) on success.
        """
        if not self._config.enabled:
            return

        if not frames:
            return

        def _encode():
            fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=self._temp_dir)
            os.close(fd)
            try:
                success = self.encode_clip(frames, tmp_path)
                if success and callback:
                    callback(event_id, tmp_path)
                elif not success:
                    os.unlink(tmp_path)
            except Exception:
                logger.exception("Error in async clip encoding for %s", event_id)
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        thread = threading.Thread(target=_encode, daemon=True)
        thread.start()
