# BirdFeeder

A single-container bird species identifier that watches an RTSP camera feed, detects birds using ML, classifies them to species, and logs every visit with snapshots, thumbnails, and video clips.

Runs on any Docker host. Optionally accelerated with a Google Coral Edge TPU.

## Features

- **Object detection** with EfficientDet-Lite0 (COCO) -- finds birds in the camera frame
- **Species classification** with Google AIY Birds V1 -- identifies 965 bird species
- **Event tracking** -- groups consecutive detections into visit events using IoU matching
- **Motion pre-filter** -- skips idle frames to reduce CPU usage
- **Video clips** -- records short MP4 clips of each bird visit from a rolling frame buffer
- **Snapshots & thumbnails** -- saves a full-frame JPEG and cropped bird thumbnail per event
- **Web UI** -- browse detections by date, hour, or species with a media viewer
- **MQTT + Home Assistant** -- publishes detections with auto-discovery for HA sensors and automations
- **Coral TPU support** -- automatically detects USB/PCIe Coral hardware and uses Edge TPU models
- **Auto-cleanup** -- deletes media older than a configurable retention period

## Quick Start

1. Copy the example config and fill in your camera URL:

```bash
mkdir -p config
cp config.example.yml config/config.yml
```

Edit `config/config.yml` and set your RTSP stream URL:

```yaml
camera:
  rtsp_url: rtsp://username:password@192.168.1.100:7447/stream
```

2. Start the container:

```bash
docker compose up -d
```

3. Open the web UI at `http://<your-host>:7766`

That's it. Media is stored locally in `./media/`, the database in `./data/`. No NAS or network storage required.

## Hardware

### Minimum (CPU only)
- 2+ CPU cores (ARM or x86_64)
- 1-2 GB RAM
- Network access to an RTSP camera

### Recommended
- Google Coral USB or PCIe Edge TPU
- SSD storage (helps with video encoding)

The Coral TPU is detected automatically at startup. If present, Edge TPU-compiled models are used for both detection and classification. If not, inference falls back to CPU.

USB passthrough is configured in `docker-compose.yml`:

```yaml
devices:
  - /dev/bus/usb:/dev/bus/usb
```

Remove this line if you don't have a Coral TPU and want to suppress the Docker warning.

## Configuration

All configuration lives in `config/config.yml`. See [`config.example.yml`](config.example.yml) for the full template with comments.

### Camera

| Key | Default | Description |
|-----|---------|-------------|
| `rtsp_url` | *(required)* | RTSP stream URL with credentials |
| `detection_fps` | `2` | Frames per second to run ML inference (1-5 typical) |
| `buffer_seconds` | `60` | Rolling frame buffer for video clip extraction |

### Motion Pre-Filter

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Skip ML inference on frames with no motion |
| `min_area` | `500` | Minimum contour area (pixels) to count as motion |

### Detection

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `efficientdet_lite0.tflite` | CPU detection model |
| `edgetpu_model` | `efficientdet_lite0_edgetpu.tflite` | Coral TPU detection model |
| `bird_confidence` | `0.5` | Minimum confidence to consider a detection a bird |

### Classification

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `bird_classifier.tflite` | CPU classification model |
| `edgetpu_model` | `bird_classifier_edgetpu.tflite` | Coral TPU classification model |
| `threshold` | `0.7` | Minimum confidence to accept a species ID |

### Tracker

| Key | Default | Description |
|-----|---------|-------------|
| `max_missing_frames` | `10` | Consecutive frames without a bird before ending the event (~5s at 2 FPS) |
| `iou_threshold` | `0.3` | Minimum bounding box overlap to match a detection to a tracked bird |

### Storage

| Key | Default | Description |
|-----|---------|-------------|
| `media_dir` | `/media` | Container path for snapshots, thumbnails, and clips |
| `data_dir` | `/data` | Container path for SQLite database |
| `retention_days` | `30` | Auto-delete media older than this many days |
| `snapshot_quality` | `95` | JPEG compression quality (1-100) |

### Video Clips

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Record video clips of bird visits |
| `pre_capture` | `3` | Seconds of video to include before detection |
| `post_capture` | `2` | Seconds of video to include after the bird leaves |
| `fps` | `15` | Output clip framerate |
| `max_duration` | `60` | Maximum clip length in seconds |

### MQTT (optional)

Omit the `mqtt` section entirely to disable MQTT. When enabled, BirdFeeder publishes detection events and integrates with Home Assistant via MQTT Discovery.

| Key | Default | Description |
|-----|---------|-------------|
| `broker` | | MQTT broker IP or hostname |
| `port` | `1883` | MQTT broker port |
| `username` | `""` | MQTT username (empty for no auth) |
| `password` | `""` | MQTT password |
| `topic_prefix` | `birdfeeder` | Prefix for all MQTT topics |
| `homeassistant_discovery` | `true` | Register sensors in Home Assistant automatically |

**Topics published:**

- `birdfeeder/detection` -- JSON payload per event (species, confidence, snapshot URL, duration)
- `birdfeeder/last_bird` -- retained state of the most recent detection
- `birdfeeder/status` -- `online` / `offline` (via MQTT last-will)

### Web UI

| Key | Default | Description |
|-----|---------|-------------|
| `port` | `7766` | Web server port |
| `host` | `0.0.0.0` | Web server bind address |

## Web UI

The web interface is available at `http://<host>:7766` and provides:

- **Dashboard** -- today's recent detections and hourly summary table
- **Date navigation** -- browse any past date with detections
- **Hourly drilldown** -- all detections in a specific hour
- **Species drilldown** -- all detections of one species on a date
- **Media viewer** -- click any thumbnail to view the full snapshot or video clip

### API

`GET /api/status` returns:

```json
{
  "status": "running",
  "events_today": 42,
  "species_today": 8
}
```

## Storage

### Default: Local Storage

By default, all media is stored locally on the Docker host:

```
./config/       -> /config    (configuration)
./data/         -> /data      (SQLite database)
./media/        -> /media     (snapshots, thumbnails, clips)
```

Media is organized by date:

```
media/snapshots/2025-02-04/
  ├── {event_id}_snapshot.jpg
  ├── {event_id}_thumb.jpg
  └── {event_id}_clip.mp4
```

### Optional: NAS Storage

To store media on a NAS instead of locally, edit `docker-compose.yml`:

1. Replace `./media:/media` with `nas_media:/media` in the service volumes
2. Uncomment the `volumes:` section at the bottom and fill in your NAS credentials

```yaml
services:
  birdfeeder:
    volumes:
      - ./config:/config
      - ./data:/data
      - nas_media:/media       # <-- use named volume instead of local

volumes:
  nas_media:
    driver: local
    driver_opts:
      type: cifs
      device: "//192.168.1.50/birdfeeder"
      o: "addr=192.168.1.50,username=user,password=pass,vers=3.0,uid=1000,gid=1000"
```

The application code is storage-agnostic -- it reads and writes to `/media` regardless of what backs it.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BIRDFEEDER_CONFIG` | `/config/config.yml` | Path to configuration file inside the container |
| `TZ` | `America/New_York` | Timezone for detection timestamps |

## Project Structure

```
birdfeeder/
├── app/
│   ├── main.py              # Entry point, component orchestration
│   ├── pipeline.py          # Detection/classification/tracking loop
│   ├── camera.py            # RTSP capture with rolling frame buffer
│   ├── detector.py          # EfficientDet-Lite0 bird detection
│   ├── classifier.py        # AIY Birds V1 species classification
│   ├── tracker.py           # IoU-based event tracking
│   ├── motion.py            # MOG2 motion pre-filter
│   ├── edgetpu.py           # Coral TPU detection and initialization
│   ├── config.py            # Configuration dataclasses and loading
│   ├── db.py                # SQLite database operations
│   ├── storage.py           # Image/video file management
│   ├── clip.py              # Async video clip encoding (ffmpeg)
│   ├── mqtt.py              # MQTT publishing and HA discovery
│   └── web/
│       ├── app.py           # Flask app factory
│       ├── routes.py        # Web routes and API
│       ├── filters.py       # Jinja2 template filters
│       ├── static/style.css
│       └── templates/
├── models/
│   └── birdnames.db         # Scientific-to-common name lookup
├── scripts/
│   └── build_birdnames_db.py
├── Dockerfile
├── docker-compose.yml
├── config.example.yml
└── requirements.txt
```

## Models

All models are downloaded during the Docker build. No manual downloads needed.

| Model | Purpose | Source |
|-------|---------|--------|
| EfficientDet-Lite0 | Object detection (COCO) | TF Hub |
| EfficientDet-Lite0 (Edge TPU) | Object detection (Coral) | google-coral/test_data |
| AIY Birds V1 | Species classification (965 species) | TF Hub |
| COCO labels | Class names for detection | google-coral/test_data |
| AIY Birds V1 labelmap | Species label mapping | Google AI Hub |
