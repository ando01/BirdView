FROM python:3.9-slim-bullseye

WORKDIR /app

# System dependencies for OpenCV, video encoding, and Coral TPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libusb-1.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Add Google Coral Edge TPU repository
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | tee /etc/apt/sources.list.d/coral.list \
    && curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get install -y --no-install-recommends libedgetpu1-std \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pycoral and tflite-runtime from Coral repo
RUN pip install --no-cache-dir \
    --extra-index-url https://google-coral.github.io/py-repo/ \
    pycoral~=2.0 \
    tflite-runtime~=2.5

# Create directories
RUN mkdir -p /config /data /media /models /tmp/clips

# Download models
# EfficientDet-Lite0 - COCO object detection (CPU)
RUN curl -L -o /models/efficientdet_lite0.tflite \
    "https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1.tflite"

# EfficientDet-Lite0 - Edge TPU compiled
RUN curl -L -o /models/efficientdet_lite0_edgetpu.tflite \
    "https://raw.githubusercontent.com/google-coral/test_data/master/efficientdet_lite0_320_ptq_edgetpu.tflite"

# Google Birds V1 classifier (CPU)
RUN curl -L -o /models/bird_classifier.tflite \
    "https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3?lite-format=tflite"

# Bird species label map
RUN curl -L -o /models/aiy_birds_V1_labelmap.csv \
    "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv"

# COCO labels
RUN curl -L -o /models/coco_labels.txt \
    "https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt"

# Copy application code
COPY app/ ./app/

# Copy birdnames database (if available locally; otherwise built at first run)
COPY models/birdnames.db /models/birdnames.db

EXPOSE 7766

VOLUME ["/config", "/data", "/media"]

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "app.main"]
