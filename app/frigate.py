import json
import logging
import threading
import time
from typing import Callable, Optional

import requests
import paho.mqtt.client as mqtt

from app.config import FrigateConfig, MQTTConfig

logger = logging.getLogger(__name__)


class FrigateClient:
    def __init__(self, config: FrigateConfig):
        self._base_url = f"http://{config.host}:{config.port}"
        self._timeout = config.api_timeout

    def get_snapshot_bytes(self, event_id: str, retries: int = 2) -> Optional[bytes]:
        url = f"{self._base_url}/api/events/{event_id}/snapshot.jpg"
        for attempt in range(1 + retries):
            try:
                resp = requests.get(url, timeout=self._timeout)
                resp.raise_for_status()
                return resp.content
            except requests.RequestException as e:
                if attempt < retries:
                    delay = (attempt + 1)  # 1s, 2s
                    logger.warning(
                        "Snapshot download attempt %d/%d failed for %s: %s — retrying in %ds",
                        attempt + 1, 1 + retries, event_id[:8], e, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.warning("Failed to download snapshot for %s after %d attempts: %s",
                                   event_id[:8], 1 + retries, e)
        return None

    def get_mjpeg_url(self, camera_name: str) -> str:
        return f"{self._base_url}/api/{camera_name}"

    def get_latest_jpg_url(self, camera_name: str) -> str:
        return f"{self._base_url}/api/{camera_name}/latest.jpg"


class FrigateConsumer(threading.Thread):
    def __init__(self, frigate_config: FrigateConfig, mqtt_config: MQTTConfig,
                 on_bird_event: Callable[[dict, Optional[Callable[[bool], None]]], None]):
        super().__init__(daemon=True, name="FrigateConsumer")
        self._frigate_cfg = frigate_config
        self._on_bird_event = on_bird_event
        self._running = True
        self._connected = False
        self._lock = threading.Lock()
        self._classified_events: set = set()  # event IDs successfully classified
        self._attempted_events: set = set()  # event IDs we've already tried (prevents update spam)

        client_id = f"birdfeeder-sub-{int(time.time())}"
        self._client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        if mqtt_config.username:
            self._client.username_pw_set(mqtt_config.username, mqtt_config.password)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._mqtt_config = mqtt_config

    @property
    def connected(self) -> bool:
        return self._connected

    def run(self):
        self._client.connect_async(self._mqtt_config.broker, self._mqtt_config.port)
        self._client.loop_start()
        while self._running:
            time.sleep(1)

    def stop(self):
        self._running = False
        self._client.loop_stop()
        self._client.disconnect()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            client.subscribe("frigate/events", qos=1)
            logger.info("FrigateConsumer connected, subscribed to frigate/events")
        else:
            logger.warning("FrigateConsumer MQTT connect failed rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning("FrigateConsumer MQTT disconnected rc=%d", rc)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        event_type = payload.get("type")
        after = payload.get("after", {})
        camera = after.get("camera", "")
        label = after.get("label", "")

        # Filter by camera
        watched = self._frigate_cfg.cameras
        if watched and camera not in watched:
            logger.debug("Ignoring event from unwatched camera %s", camera)
            return
        if label != "bird":
            logger.debug("Ignoring non-bird label %r on %s", label, camera)
            return

        event_id = after.get("id")
        if not event_id:
            logger.debug("Ignoring bird event with no id on %s", camera)
            return

        # Decide whether to process this message
        already_classified = event_id in self._classified_events
        has_snapshot = after.get("has_snapshot", False)

        already_attempted = event_id in self._attempted_events

        if event_type in ("new", "update") and has_snapshot and not already_attempted:
            # Process eagerly on first snapshot — bird is in frame now
            self._attempted_events.add(event_id)
            logger.info(
                "Bird event (early): id=%s camera=%s type=%s score=%.2f",
                event_id[:8], camera, event_type, after.get("score", 0.0),
            )
        elif event_type == "end" and not already_classified:
            # Fallback: process on end if we haven't classified yet
            logger.info(
                "Bird event (end fallback): id=%s camera=%s score=%.2f",
                event_id[:8], camera, after.get("score", 0.0),
            )
        elif event_type == "end" and already_classified:
            # Still process — clip download only happens on "end" (needs end_time)
            logger.info(
                "Bird event (end, clip pass): id=%s camera=%s score=%.2f",
                event_id[:8], camera, after.get("score", 0.0),
            )
        else:
            logger.debug(
                "Skipping bird event id=%s type=%s has_snapshot=%s already_classified=%s",
                event_id[:8], event_type, has_snapshot, already_classified,
            )
            return

        normalized = {
            "event_id": event_id,
            "event_type": event_type,
            "camera": camera,
            "frigate_score": after.get("score", 0.0),
            "box": after.get("box", []),
            "start_time": after.get("start_time"),
            "end_time": after.get("end_time"),
            "has_snapshot": has_snapshot,
            "has_clip": after.get("has_clip", False),
        }

        def _on_result(success: bool):
            if success:
                self._classified_events.add(event_id)
                # Cap set sizes to prevent unbounded growth
                if len(self._classified_events) > 500:
                    self._classified_events.clear()
            if len(self._attempted_events) > 500:
                self._attempted_events.clear()

        try:
            self._on_bird_event(normalized, _on_result)
        except Exception:
            logger.exception("Error dispatching Frigate event %s", event_id[:8])
