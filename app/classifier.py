import csv
import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.config import ClassificationConfig
from app.edgetpu import create_interpreter

logger = logging.getLogger(__name__)

MODELS_DIR = "/models"
BACKGROUND_INDEX = 964
LABELS_CSV = "aiy_birds_V1_labelmap.csv"
LABELS_TXT = "inat_bird_labels.txt"
BIRDNAMES_DB = "birdnames.db"


@dataclass
class Classification:
    scientific_name: str
    common_name: str
    score: float
    label_index: int


class BirdClassifier:
    def __init__(self, config: ClassificationConfig, db=None):
        cpu_path = os.path.join(MODELS_DIR, config.model)
        edgetpu_path = os.path.join(MODELS_DIR, config.edgetpu_model)

        self._interpreter, self._using_edgetpu = create_interpreter(cpu_path, edgetpu_path)
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._input_size = config.input_size
        self._threshold = config.threshold
        self._db = db
        self._config = config

        self._labels = self._load_labels()
        self._common_names = {}  # Maps scientific name to common name (for iNaturalist)
        self._birdnames_db = os.path.join(MODELS_DIR, BIRDNAMES_DB)

        logger.info(
            "BirdClassifier ready (Edge TPU: %s, labels: %d species)",
            self._using_edgetpu, len(self._labels),
        )

    @property
    def using_edgetpu(self) -> bool:
        return self._using_edgetpu

    def _load_labels(self) -> dict:
        """Load label map from CSV or TXT. Returns {index: scientific_name}."""
        labels = {}

        # Try iNaturalist TXT format first (for Edge TPU model)
        txt_path = os.path.join(MODELS_DIR, LABELS_TXT)
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if line:
                        # Format: "Scientific Name (Common Name)"
                        if '(' in line and ')' in line:
                            scientific = line.split('(')[0].strip()
                            common = line.split('(')[1].split(')')[0].strip()
                            labels[idx] = scientific
                            self._common_names[scientific] = common
                        else:
                            labels[idx] = line
            logger.info("Loaded %d labels from %s (iNaturalist)", len(labels), LABELS_TXT)
            return labels

        # Fallback to AIY Birds V1 CSV format
        csv_path = os.path.join(MODELS_DIR, LABELS_CSV)
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) >= 2:
                        idx = int(row[0])
                        name = row[1].strip()
                        labels[idx] = name
            logger.info("Loaded %d labels from %s (AIY Birds V1)", len(labels), LABELS_CSV)
        else:
            logger.warning("Label map not found: %s or %s", txt_path, csv_path)

        return labels

    def _get_common_name(self, scientific_name: str) -> str:
        # First check if we have it from iNaturalist labels
        if scientific_name in self._common_names:
            return self._common_names[scientific_name]

        # Fallback to birdnames database (for AIY Birds V1)
        if not os.path.exists(self._birdnames_db):
            return scientific_name
        try:
            conn = sqlite3.connect(self._birdnames_db)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT common_name FROM birdnames WHERE scientific_name = ?",
                (scientific_name,),
            )
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else scientific_name
        except Exception:
            return scientific_name

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize with aspect ratio preservation, pad to input_size with black."""
        target = self._input_size
        h, w = image.shape[:2]

        if h == 0 or w == 0:
            return np.zeros((target, target, 3), dtype=np.uint8)

        scale = min(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        y_off = (target - new_h) // 2
        x_off = (target - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        return canvas

    def classify(self, bird_crop: np.ndarray) -> Optional[Classification]:
        """Classify a cropped bird image. Returns None if below threshold."""
        # Get dynamic threshold from database
        threshold = self._threshold
        if self._db:
            threshold = self._db.get_setting("classification_threshold", self._threshold)

        processed = self._preprocess(bird_crop)
        input_data = np.expand_dims(processed, axis=0).astype(np.uint8)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()

        output = self._interpreter.get_tensor(self._output_details[0]["index"])[0]

        # Handle quantized uint8 output
        output_dtype = self._output_details[0]["dtype"]
        if output_dtype == np.uint8:
            scores = output.astype(np.float32) / 255.0
        else:
            scores = output.astype(np.float32)

        top_index = int(np.argmax(scores))
        top_score = float(scores[top_index])

        if top_index == BACKGROUND_INDEX:
            return None
        if top_score < threshold:
            return None

        scientific_name = self._labels.get(top_index, f"Unknown ({top_index})")
        common_name = self._get_common_name(scientific_name)

        return Classification(
            scientific_name=scientific_name,
            common_name=common_name,
            score=top_score,
            label_index=top_index,
        )
