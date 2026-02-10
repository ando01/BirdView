import logging
import os
import subprocess
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

    def _transcode_to_h264(self, input_path: str, output_path: str) -> bool:
        """Transcode video to H.264 using ffmpeg for browser compatibility."""
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.error("ffmpeg transcode failed: %s", result.stderr.decode())
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg transcode timed out for %s", input_path)
            return False
        except Exception:
            logger.exception("Error transcoding clip")
            return False

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

        # First encode with OpenCV (mp4v codec)
        temp_raw = output_path + ".raw.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_raw, fourcc, fps, (w, h))

        if not writer.isOpened():
            logger.error("Failed to open VideoWriter for %s", temp_raw)
            return False

        try:
            for _, frame in frames:
                writer.write(frame)
        except Exception:
            logger.exception("Error encoding clip")
            return False
        finally:
            writer.release()

        # Transcode to H.264 for browser compatibility
        success = self._transcode_to_h264(temp_raw, output_path)

        # Clean up temporary file
        if os.path.exists(temp_raw):
            os.unlink(temp_raw)

        return success

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
