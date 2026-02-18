import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ClassificationConfig:
    model: str = "bird_classifier.tflite"
    edgetpu_model: str = "bird_classifier_edgetpu.tflite"
    threshold: float = 0.7
    input_size: int = 224


@dataclass
class FrigateConfig:
    host: str = "localhost"
    port: int = 5000
    cameras: list = field(default_factory=list)  # empty = all cameras
    api_timeout: int = 10
    process_on_snapshot: bool = False  # True = classify on first snapshot, not event end
    download_clips: bool = True
    clip_download_delay: int = 5       # seconds to wait before fetching clip


@dataclass
class StorageConfig:
    media_dir: str = "/media"
    data_dir: str = "/data"
    retention_days: int = 30
    snapshot_quality: int = 95


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
    frigate: FrigateConfig
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
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

    frigate_raw = raw.get("frigate", {})
    if not frigate_raw or not frigate_raw.get("host"):
        raise ValueError("frigate.host is required in config.yml")
    frigate = _build_dataclass(FrigateConfig, frigate_raw)

    classification = _build_dataclass(ClassificationConfig, raw.get("classification"))
    storage = _build_dataclass(StorageConfig, raw.get("storage"))
    web = _build_dataclass(WebConfig, raw.get("web"))
    logging_cfg = _build_dataclass(LoggingConfig, raw.get("logging"))

    mqtt = None
    if "mqtt" in raw and raw["mqtt"]:
        mqtt_raw = raw["mqtt"]
        if mqtt_raw.get("broker"):
            mqtt = _build_dataclass(MQTTConfig, mqtt_raw)

    config = AppConfig(
        frigate=frigate,
        classification=classification,
        storage=storage,
        mqtt=mqtt,
        web=web,
        logging=logging_cfg,
    )

    logger.info("Configuration loaded from %s", path)
    logger.info("  Frigate host: %s:%d", config.frigate.host, config.frigate.port)
    logger.info("  Frigate cameras: %s", config.frigate.cameras or "all")
    logger.info("  Classifier model: %s", config.classification.edgetpu_model)
    logger.info("  MQTT: %s", "enabled" if config.mqtt else "disabled")
    logger.info("  Media dir: %s", config.storage.media_dir)

    return config
