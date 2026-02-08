import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    rtsp_url: str
    detection_fps: float = 2.0
    buffer_seconds: int = 60
    reconnect_delay: float = 5.0
    max_reconnect_delay: float = 60.0


@dataclass
class MotionConfig:
    enabled: bool = True
    min_area: int = 500
    history: int = 50


@dataclass
class DetectionConfig:
    model: str = "efficientdet_lite0.tflite"
    edgetpu_model: str = "efficientdet_lite0_edgetpu.tflite"
    bird_confidence: float = 0.5
    input_size: int = 320


@dataclass
class ClassificationConfig:
    model: str = "bird_classifier.tflite"
    edgetpu_model: str = "bird_classifier_edgetpu.tflite"
    threshold: float = 0.7
    input_size: int = 224


@dataclass
class TrackerConfig:
    max_missing_frames: int = 10
    iou_threshold: float = 0.3


@dataclass
class StorageConfig:
    media_dir: str = "/media"
    data_dir: str = "/data"
    retention_days: int = 30
    snapshot_quality: int = 95


@dataclass
class ClipConfig:
    enabled: bool = True
    pre_capture: int = 3
    post_capture: int = 2
    fps: int = 15
    max_duration: int = 60


@dataclass
class MQTTConfig:
    broker: str = ""
    port: int = 1883
    username: str = ""
    password: str = ""
    topic_prefix: str = "birdfeeder"
    homeassistant_discovery: bool = True


@dataclass
class WebConfig:
    port: int = 7766
    host: str = "0.0.0.0"


@dataclass
class LoggingConfig:
    filter_internal_ips: bool = True


@dataclass
class AppConfig:
    camera: CameraConfig
    motion: MotionConfig = field(default_factory=MotionConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    clips: ClipConfig = field(default_factory=ClipConfig)
    mqtt: Optional[MQTTConfig] = None
    web: WebConfig = field(default_factory=WebConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _build_dataclass(cls, data: dict):
    """Build a dataclass from a dict, ignoring unknown keys."""
    if data is None:
        return cls()
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


def load_config(path: str = "/config/config.yml") -> AppConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    camera_raw = raw.get("camera", {})
    if not camera_raw or not camera_raw.get("rtsp_url"):
        raise ValueError("camera.rtsp_url is required in config.yml")

    camera = _build_dataclass(CameraConfig, camera_raw)
    motion = _build_dataclass(MotionConfig, raw.get("motion"))
    detection = _build_dataclass(DetectionConfig, raw.get("detection"))
    classification = _build_dataclass(ClassificationConfig, raw.get("classification"))
    tracker = _build_dataclass(TrackerConfig, raw.get("tracker"))
    storage = _build_dataclass(StorageConfig, raw.get("storage"))
    clips = _build_dataclass(ClipConfig, raw.get("clips"))
    web = _build_dataclass(WebConfig, raw.get("web"))
    logging_cfg = _build_dataclass(LoggingConfig, raw.get("logging"))

    mqtt = None
    if "mqtt" in raw and raw["mqtt"]:
        mqtt_raw = raw["mqtt"]
        if mqtt_raw.get("broker"):
            mqtt = _build_dataclass(MQTTConfig, mqtt_raw)

    config = AppConfig(
        camera=camera,
        motion=motion,
        detection=detection,
        classification=classification,
        tracker=tracker,
        storage=storage,
        clips=clips,
        mqtt=mqtt,
        web=web,
        logging=logging_cfg,
    )

    logger.info("Configuration loaded from %s", path)
    logger.info("  RTSP URL: %s", _redact_url(config.camera.rtsp_url))
    logger.info("  Detection FPS: %s", config.camera.detection_fps)
    logger.info("  Coral TPU models: %s / %s", config.detection.edgetpu_model, config.classification.edgetpu_model)
    logger.info("  MQTT: %s", "enabled" if config.mqtt else "disabled")
    logger.info("  Clips: %s", "enabled" if config.clips.enabled else "disabled")
    logger.info("  Media dir: %s", config.storage.media_dir)

    return config


def _redact_url(url: str) -> str:
    """Redact password from RTSP URL for logging."""
    if "@" in url:
        prefix, rest = url.split("@", 1)
        if ":" in prefix:
            scheme_user = prefix.rsplit(":", 1)[0]
            return f"{scheme_user}:****@{rest}"
    return url
