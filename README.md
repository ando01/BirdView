# BirdFeeder

A single-container bird species identifier that integrates with [Frigate NVR](https://frigate.video). When Frigate detects a bird, BirdFeeder pulls the snapshot, classifies it to species using a TensorFlow Lite model, and logs every visit with snapshots, thumbnails, and video clips.

Runs on any Docker host on the same network as Frigate. Optionally accelerated with a Google Coral Edge TPU.

## Features

- **Frigate integration** -- subscribes to Frigate MQTT events; no direct camera access needed
- **Species classification** -- identifies 965+ bird species using iNaturalist/Google AIY models
- **Coral TPU support** -- automatically uses Edge TPU models when a USB/PCIe Coral is detected
- **Snapshots & thumbnails** -- saves full-frame JPEG and cropped bird thumbnail per event
- **Video clips** -- downloads MP4 clips from Frigate after each event
- **Web UI** -- browse detections by date, hour, or species with a media viewer
- **MQTT + Home Assistant** -- publishes detections with auto-discovery for HA sensors
- **Auto-cleanup** -- deletes media older than a configurable retention period

## Quick Start

BirdFeeder is published as a pre-built image on GitHub Container Registry. No build step required.

### 1. Get the compose file and example config

```bash
mkdir birdfeeder && cd birdfeeder
curl -O https://raw.githubusercontent.com/ando01/BirdView/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/ando01/BirdView/main/config.example.yml
mkdir config
cp config.example.yml config/config.yml
```

### 2. Edit config/config.yml

Set your Frigate server address and MQTT broker:

```yaml
frigate:
  host: 192.168.1.x    # IP or hostname of your Frigate server
  port: 5000
  cameras:
    - backyard           # camera name(s) from your Frigate config

mqtt:
  broker: 192.168.1.x  # usually the same host as Frigate / Home Assistant
  port: 1883
```

### 3. Start the container

```bash
# Static IP (recommended — container gets a fixed LAN IP):
docker compose --profile static up -d

# DHCP (container requests an IP from your router):
docker compose --profile dhcp up -d
```

> **Network note:** The compose file uses a `macvlan` network so BirdFeeder gets its own IP address directly on your LAN. This lets it reach Frigate and your MQTT broker at their normal LAN addresses with no extra routing.

### 4. Open the web UI

`http://<container-ip>:7766`

## docker-compose.yml setup

Before starting, set the values marked with `<--` in `docker-compose.yml`:

| Setting | Location | Description |
|---------|----------|-------------|
| `ipv4_address` | `lan_static` network | Fixed IP to assign the container (static profile) |
| `parent` | both networks | Host network interface (e.g. `eth0`, `eno1`, `enp3s0`) |
| `subnet` / `gateway` | `lan_static` network | Your LAN subnet and router IP |

Remove the `devices:` entry if you don't have a Coral TPU.

## Configuration

All configuration lives in `config/config.yml`. See [`config.example.yml`](config.example.yml) for the full template with comments.

### Frigate

| Key | Default | Description |
|-----|---------|-------------|
| `host` | *(required)* | Frigate server IP or hostname |
| `port` | `5000` | Frigate web/API port |
| `cameras` | *(all)* | Camera names to watch (empty list = all cameras) |
| `api_timeout` | `10` | Seconds before Frigate API requests time out |
| `process_on_snapshot` | `false` | Classify on first snapshot instead of waiting for event end |
| `download_clips` | `true` | Download MP4 clips from Frigate after each event |
| `clip_download_delay` | `5` | Seconds to wait before fetching the clip (lets Frigate finalize it) |

### Classification

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `bird_classifier.tflite` | CPU classification model |
| `edgetpu_model` | `bird_classifier_edgetpu.tflite` | Coral TPU model (used automatically when TPU detected) |
| `threshold` | `0.7` | Minimum confidence to accept a species identification |

### Storage

| Key | Default | Description |
|-----|---------|-------------|
| `media_dir` | `/media` | Container path for snapshots, thumbnails, and clips |
| `data_dir` | `/data` | Container path for SQLite database |
| `retention_days` | `30` | Auto-delete media older than this many days |
| `snapshot_quality` | `95` | JPEG compression quality (1-100) |

### MQTT

Omit the `mqtt` section entirely to disable MQTT. MQTT is required for receiving Frigate events.

| Key | Default | Description |
|-----|---------|-------------|
| `broker` | *(required)* | MQTT broker IP or hostname |
| `port` | `1883` | MQTT broker port |
| `username` | `""` | MQTT username (empty = no auth) |
| `password` | `""` | MQTT password |
| `topic_prefix` | `birdfeeder` | Prefix for all published topics |
| `homeassistant_discovery` | `true` | Auto-register sensors in Home Assistant |

**Topics published:**

- `birdfeeder/detection` -- JSON payload per event (species, confidence, snapshot URL, duration)
- `birdfeeder/last_bird` -- retained state of the most recent detection
- `birdfeeder/status` -- `online` / `offline` (via MQTT last-will)

### Web

| Key | Default | Description |
|-----|---------|-------------|
| `port` | `7766` | Web server port |
| `host` | `0.0.0.0` | Web server bind address |

## NAS Storage (optional)

To store media on a NAS instead of locally, edit `docker-compose.yml`:

1. Replace `./media:/media` with `nas_media:/media` in the service volumes
2. Uncomment the `volumes:` section at the bottom and fill in your NAS details

```yaml
volumes:
  nas_media:
    driver: local
    driver_opts:
      type: cifs
      device: "//192.168.1.50/birdfeeder"
      o: "addr=192.168.1.50,username=user,password=pass,vers=3.0,uid=1000,gid=1000"
```

## Hardware

### Minimum (CPU only)
- 2+ CPU cores (ARM or x86_64)
- 1 GB RAM
- Network access to your Frigate server and MQTT broker

### Recommended
- Google Coral USB or PCIe Edge TPU — significantly faster classification

The Coral TPU is detected automatically at startup. If present, the Edge TPU-compiled model is used. If not, inference falls back to CPU.

USB passthrough is configured in `docker-compose.yml`:

```yaml
devices:
  - /dev/bus/usb:/dev/bus/usb
```

Remove this line if you don't have a Coral TPU.

## Models

All models are downloaded during the Docker build. No manual downloads needed.

| Model | Purpose | Source |
|-------|---------|--------|
| iNaturalist Bird (Edge TPU) | Species classification — primary | google-coral/test_data |
| Google AIY Birds V1 | Species classification — CPU fallback | TF Hub |
| iNaturalist bird labels | 965+ species label map | google-coral/test_data |
| AIY Birds V1 labelmap | Label map for fallback model | Google AI Hub |

## Web UI

Available at `http://<container-ip>:7766`:

- **Dashboard** -- today's recent detections and hourly summary
- **Date navigation** -- browse any past date
- **Species drilldown** -- all detections of a species on a given date
- **Media viewer** -- full snapshot or video clip for each event

### API

`GET /api/status`

```json
{ "status": "running", "events_today": 42, "species_today": 8 }
```

## Project Structure

```
birdfeeder/
├── app/
│   ├── main.py          # Entry point and component orchestration
│   ├── pipeline.py      # Event processing pipeline
│   ├── frigate.py       # Frigate MQTT consumer and HTTP client
│   ├── classifier.py    # TFLite species classification
│   ├── config.py        # Configuration dataclasses and loading
│   ├── db.py            # SQLite database operations
│   ├── storage.py       # Image/video file management
│   ├── mqtt.py          # MQTT publishing and HA discovery
│   └── web/
│       ├── app.py       # Flask app factory
│       ├── routes.py    # Web routes and API endpoints
│       └── templates/
├── models/
│   └── birdnames.db     # Scientific-to-common name lookup
├── .github/workflows/
│   └── docker-publish.yml  # Auto-build and push to ghcr.io on push to main
├── Dockerfile
├── docker-compose.yml
├── config.example.yml
└── requirements.txt
```
