import ipaddress
import logging
import os
import re
import signal
import sys
import threading
import time

from app.camera import CameraStream
from app.classifier import BirdClassifier
from app.clip import ClipEncoder
from app.config import load_config
from app.db import Database
from app.detector import BirdDetector
from app.log_buffer import log_buffer
from app.motion import MotionDetector
from app.mqtt import MQTTPublisher
from app.pipeline import DetectionPipeline
from app.storage import ImageStorage
from app.tracker import EventTracker
from app.web.app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("birdfeeder")


class WerkzeugInternalIPFilter(logging.Filter):
    """Filter out werkzeug request logs from internal/private IP addresses."""

    IP_PATTERN = re.compile(r"^(\d+\.\d+\.\d+\.\d+)")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        match = self.IP_PATTERN.match(message)
        if match:
            try:
                ip = ipaddress.ip_address(match.group(1))
                return not ip.is_private
            except ValueError:
                pass
        return True


werkzeug_filter = WerkzeugInternalIPFilter()

# Add log buffer handler to capture logs for web UI
log_buffer.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(log_buffer)


def cleanup_scheduler(db: Database, storage: ImageStorage, retention_days: int):
    """Hourly cleanup of old detections and media."""
    while True:
        time.sleep(3600)
        try:
            count = db.cleanup_old_detections(retention_days)
            if count:
                logger.info("Cleaned up %d old detections from database", count)
            storage.cleanup_old(retention_days)
        except Exception:
            logger.exception("Error during cleanup")


def main():
    config_path = os.environ.get("BIRDFEEDER_CONFIG", "/config/config.yml")

    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    # Configure werkzeug log filtering
    werkzeug_logger = logging.getLogger("werkzeug")
    if config.logging.filter_internal_ips:
        werkzeug_logger.addFilter(werkzeug_filter)
        logger.info("Filtering werkzeug logs from internal IPs")

    # Database
    db_path = os.path.join(config.storage.data_dir, "birdfeeder.db")
    db = Database(db_path)

    # Media storage
    storage = ImageStorage(config.storage.media_dir, config.storage.snapshot_quality)

    # ML models
    try:
        detector = BirdDetector(config.detection)
        classifier = BirdClassifier(config.classification)
    except Exception as e:
        logger.error("Failed to load ML models: %s", e)
        sys.exit(1)

    # Camera
    camera = CameraStream(config.camera)

    # Motion detector
    motion = MotionDetector(config.motion) if config.motion.enabled else None

    # Event tracker
    tracker = EventTracker(config.tracker)

    # Clip encoder
    clip_encoder = None
    if config.clips.enabled:
        clip_encoder = ClipEncoder(config.clips)

    # MQTT
    mqtt_pub = None
    if config.mqtt:
        mqtt_pub = MQTTPublisher(config.mqtt, config.web)

    # Event callback for MQTT
    def on_event_complete(event, snapshot_path):
        if mqtt_pub:
            mqtt_pub.publish_detection(event, snapshot_path)

    # Detection pipeline
    pipeline = DetectionPipeline(
        config=config,
        camera=camera,
        motion=motion,
        detector=detector,
        classifier=classifier,
        tracker=tracker,
        db=db,
        storage=storage,
        clip_encoder=clip_encoder,
        on_event_complete=on_event_complete,
    )

    # Start threads
    camera.start()
    pipeline.start()

    if mqtt_pub:
        mqtt_pub.start()

    # Cleanup scheduler
    cleanup_thread = threading.Thread(
        target=cleanup_scheduler,
        args=(db, storage, config.storage.retention_days),
        daemon=True,
    )
    cleanup_thread.start()

    # Graceful shutdown
    def shutdown(signum, frame):
        logger.info("Shutting down...")
        pipeline.stop()
        camera.stop()
        if mqtt_pub:
            mqtt_pub.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Flask web server (blocking, runs in main thread)
    logger.info("Starting web server on %s:%d", config.web.host, config.web.port)
    flask_app = create_app(db, storage, config, camera, pipeline, werkzeug_filter)
    flask_app.run(
        host=config.web.host,
        port=config.web.port,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
