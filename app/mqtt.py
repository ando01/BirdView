import json
import logging
import threading
from typing import Optional

import paho.mqtt.client as mqtt

from app.config import MQTTConfig, WebConfig
from app.tracker import TrackedBird

logger = logging.getLogger(__name__)


class MQTTPublisher:
    """Optional MQTT publisher for Home Assistant integration."""

    def __init__(self, config: MQTTConfig, web_config: WebConfig):
        self._config = config
        self._web_port = web_config.port
        self._client = mqtt.Client(client_id="birdfeeder", protocol=mqtt.MQTTv311)
        self._connected = False

        if config.username:
            self._client.username_pw_set(config.username, config.password)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        # Last will for availability
        self._client.will_set(
            f"{config.topic_prefix}/status",
            payload="offline",
            qos=1,
            retain=True,
        )

    def start(self):
        try:
            self._client.connect_async(self._config.broker, self._config.port)
            self._client.loop_start()
            logger.info("MQTT client connecting to %s:%d", self._config.broker, self._config.port)
        except Exception:
            logger.exception("Failed to start MQTT client")

    def stop(self):
        self._publish(f"{self._config.topic_prefix}/status", "offline", retain=True)
        self._client.loop_stop()
        self._client.disconnect()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected to broker")
            self._publish(
                f"{self._config.topic_prefix}/status", "online", retain=True
            )
            if self._config.homeassistant_discovery:
                self._publish_ha_discovery()
        else:
            logger.warning("MQTT connection failed with code %d", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning("MQTT disconnected unexpectedly (rc=%d)", rc)

    def _publish(self, topic: str, payload: str, retain: bool = False):
        if self._connected:
            self._client.publish(topic, payload, qos=1, retain=retain)

    def publish_detection(self, event: TrackedBird, snapshot_path: Optional[str]):
        """Publish a bird detection event."""
        if not self._connected:
            return

        cls = event.best_classification
        if cls is None:
            return

        duration = (event.last_seen - event.first_seen).total_seconds()
        prefix = self._config.topic_prefix

        payload = {
            "event_id": event.event_id,
            "common_name": cls.common_name,
            "scientific_name": cls.scientific_name,
            "score": round(cls.score, 3),
            "detection_time": event.first_seen.isoformat(),
            "duration_seconds": round(duration, 1),
        }

        if snapshot_path:
            payload["snapshot_url"] = (
                f"http://birdfeeder:{self._web_port}/media/{snapshot_path}"
            )

        self._publish(
            f"{prefix}/detection",
            json.dumps(payload),
        )

        # Update state topic for HA sensor
        self._publish(
            f"{prefix}/last_bird",
            json.dumps({
                "common_name": cls.common_name,
                "scientific_name": cls.scientific_name,
                "score": round(cls.score, 3),
                "time": event.first_seen.strftime("%H:%M:%S"),
            }),
            retain=True,
        )

    def _publish_ha_discovery(self):
        """Publish Home Assistant MQTT Discovery config."""
        prefix = self._config.topic_prefix

        # Last bird sensor
        self._publish(
            f"homeassistant/sensor/birdfeeder/last_bird/config",
            json.dumps({
                "name": "Bird Feeder - Last Bird",
                "unique_id": "birdfeeder_last_bird",
                "state_topic": f"{prefix}/last_bird",
                "value_template": "{{ value_json.common_name }}",
                "json_attributes_topic": f"{prefix}/last_bird",
                "icon": "mdi:bird",
                "availability_topic": f"{prefix}/status",
                "device": {
                    "identifiers": ["birdfeeder"],
                    "name": "Bird Feeder",
                    "manufacturer": "BirdFeeder",
                    "model": "v1",
                },
            }),
            retain=True,
        )

        # Detection event trigger
        self._publish(
            f"homeassistant/device_automation/birdfeeder/detection/config",
            json.dumps({
                "automation_type": "trigger",
                "type": "bird_detected",
                "subtype": "new_detection",
                "topic": f"{prefix}/detection",
                "device": {
                    "identifiers": ["birdfeeder"],
                    "name": "Bird Feeder",
                    "manufacturer": "BirdFeeder",
                    "model": "v1",
                },
            }),
            retain=True,
        )

        logger.info("Published Home Assistant MQTT Discovery config")
