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

    def get_snapshot_bytes(self, event_id: str) -> Optional[bytes]:
        url = f"{self._base_url}/api/events/{event_id}/snapshot.jpg"
        try:
            resp = requests.get(url, timeout=self._timeout)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            logger.warning("Failed to download snapshot for %s: %s", event_id[:8], e)
            return None

    def download_clip(self, event_id: str, dest_path: str) -> bool:
        url = f"{self._base_url}/api/events/{event_id}/clip.mp4"
        try:
            with requests.get(url, stream=True, timeout=self._timeout) as resp:
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
            return True
        except requests.RequestException as e:
            logger.warning("Failed to download clip for %s: %s", event_id[:8], e)
            return False

    def get_mjpeg_url(self, camera_name: str) -> str:
        return f"{self._base_url}/api/{camera_name}"

    def get_latest_jpg_url(self, camera_name: str) -> str:
        return f"{self._base_url}/api/{camera_name}/latest.jpg"


class FrigateConsumer(threading.Thread):
    def __init__(self, frigate_config: FrigateConfig, mqtt_config: MQTTConfig,
                 on_bird_event: Callable[[dict], None]):
        super().__init__(daemon=True, name="FrigateConsumer")
        self._frigate_cfg = frigate_config
        self._on_bird_event = on_bird_event
        self._running = True
        self._connected = False
        self._lock = threading.Lock()

        self._client = mqtt.Client(client_id="birdfeeder-frigate-sub", protocol=mqtt.MQTTv311)
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
            return
        if label != "bird":
            return

        # Only process on end (or eagerly on snapshot if configured)
        should_process = event_type == "end"
        if not should_process and self._frigate_cfg.process_on_snapshot:
            should_process = event_type in ("new", "update") and after.get("has_snapshot")
        if not should_process:
            return

        event_id = after.get("id")
        if not event_id:
            return

        normalized = {
            "event_id": event_id,
            "camera": camera,
            "frigate_score": after.get("score", 0.0),
            "box": after.get("box", []),
            "start_time": after.get("start_time"),
            "end_time": after.get("end_time"),
            "has_snapshot": after.get("has_snapshot", False),
            "has_clip": after.get("has_clip", False),
        }
        try:
            self._on_bird_event(normalized)
        except Exception:
            logger.exception("Error dispatching Frigate event %s", event_id[:8])
