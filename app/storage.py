import logging
import os
import shutil
from datetime import datetime, timedelta

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageStorage:
    def __init__(self, media_dir: str, snapshot_quality: int = 95):
        self._base_dir = os.path.join(media_dir, "snapshots")
        self._quality = snapshot_quality
        os.makedirs(self._base_dir, exist_ok=True)

    @property
    def base_dir(self) -> str:
        return self._base_dir

    def _date_dir(self, dt: datetime = None) -> str:
        dt = dt or datetime.now()
        d = os.path.join(self._base_dir, dt.strftime("%Y-%m-%d"))
        os.makedirs(d, exist_ok=True)
        return d

    def save_snapshot(self, event_id: str, frame: np.ndarray, dt: datetime = None) -> str:
        date_dir = self._date_dir(dt)
        filename = f"{event_id}_snapshot.jpg"
        path = os.path.join(date_dir, filename)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        return os.path.relpath(path, self._base_dir)

    def save_thumbnail(
        self, event_id: str, frame: np.ndarray, bbox: tuple, dt: datetime = None
    ) -> str:
        """Save cropped bird region as thumbnail.

        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates.
        """
        date_dir = self._date_dir(dt)
        filename = f"{event_id}_thumb.jpg"
        path = os.path.join(date_dir, filename)

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = 20
        y1c = max(0, y1 - pad)
        y2c = min(h, y2 + pad)
        x1c = max(0, x1 - pad)
        x2c = min(w, x2 + pad)
        crop = frame[y1c:y2c, x1c:x2c]

        if crop.size == 0:
            crop = frame

        th, tw = crop.shape[:2]
        if tw > 200:
            scale = 200 / tw
            crop = cv2.resize(crop, (200, int(th * scale)))

        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return os.path.relpath(path, self._base_dir)

    def save_clip(self, event_id: str, clip_path_abs: str, dt: datetime = None) -> str:
        """Move an encoded clip file into the date directory. Returns relative path."""
        date_dir = self._date_dir(dt)
        filename = f"{event_id}_clip.mp4"
        dest = os.path.join(date_dir, filename)
        shutil.move(clip_path_abs, dest)
        return os.path.relpath(dest, self._base_dir)

    def get_absolute_path(self, relative_path: str) -> str:
        return os.path.join(self._base_dir, relative_path)

    def cleanup_old(self, retention_days: int):
        cutoff = datetime.now() - timedelta(days=retention_days)
        if not os.path.exists(self._base_dir):
            return
        for dirname in os.listdir(self._base_dir):
            dirpath = os.path.join(self._base_dir, dirname)
            if not os.path.isdir(dirpath):
                continue
            try:
                dir_date = datetime.strptime(dirname, "%Y-%m-%d")
                if dir_date < cutoff:
                    shutil.rmtree(dirpath)
                    logger.info("Cleaned up media for %s", dirname)
            except ValueError:
                continue
