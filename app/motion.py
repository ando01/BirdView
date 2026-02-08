import logging

import cv2
import numpy as np

from app.config import MotionConfig

logger = logging.getLogger(__name__)


class MotionDetector:
    def __init__(self, config: MotionConfig):
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=16,
            detectShadows=False,
        )
        self._min_area = config.min_area
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        logger.info("MotionDetector ready (min_area=%d)", self._min_area)

    def has_motion(self, frame: np.ndarray) -> bool:
        """Check if the frame contains significant motion."""
        fg_mask = self._bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            max_area = max(max_area, area)
            if area > self._min_area:
                logger.debug("Motion detected: area=%d (threshold=%d)", int(area), self._min_area)
                return True

        if contours:
            logger.debug("Motion rejected: max_area=%d < threshold=%d (%d contours)",
                        int(max_area), self._min_area, len(contours))
        return False
