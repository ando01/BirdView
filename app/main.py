import ipaddress
import logging
import os
import re
import signal
import sys
import threading
import time

from app.classifier import BirdClassifier
from app.config import load_config
from app.db import Database
from app.frigate import FrigateClient, FrigateConsumer
from app.log_buffer import log_buffer
from app.mqtt import MQTTPublisher
from app.pipeline import FrigatePipeline
from app.storage import ImageStorage
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

    if not config.mqtt:
        logger.error("MQTT configuration is required for Frigate event subscription")
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

    # ML classifier
    try:
        classifier = BirdClassifier(config.classification, db=db)
    except Exception as e:
        logger.error("Failed to load classifier model: %s", e)
        sys.exit(1)

    # Frigate client
    frigate_client = FrigateClient(config.frigate)

    # MQTT publisher
    mqtt_pub = MQTTPublisher(config.mqtt, config.web)

    # Event callback for MQTT
    def on_event_complete(event_id, classification, detection_time, duration,
                          detection_confidence, snapshot_path):
        mqtt_pub.publish_detection(
            event_id=event_id,
            classification=classification,
            detection_time=detection_time,
            duration=duration,
            detection_confidence=detection_confidence,
            snapshot_path=snapshot_path,
        )

    # Frigate pipeline
    pipeline = FrigatePipeline(
        config=config,
        frigate_client=frigate_client,
        classifier=classifier,
        db=db,
        storage=storage,
        on_event_complete=on_event_complete,
    )

    # Frigate consumer (subscribes to MQTT frigate/events)
    consumer = FrigateConsumer(config.frigate, config.mqtt, pipeline.process_event)

    # Start threads
    consumer.start()
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
        consumer.stop()
        mqtt_pub.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Flask web server (blocking, runs in main thread)
    logger.info("Starting web server on %s:%d", config.web.host, config.web.port)
    flask_app = create_app(db, storage, config, consumer, pipeline, werkzeug_filter)
    flask_app.run(
        host=config.web.host,
        port=config.web.port,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
