import logging
from datetime import datetime

from flask import (
    Flask,
    current_app,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from app.log_buffer import log_buffer


def register_routes(app: Flask):

    @app.route("/")
    def index():
        db = current_app.config["db"]
        today = datetime.now().strftime("%Y-%m-%d")
        earliest = db.earliest_detection_date()
        recent = db.recent_detections(limit=10)
        summary = db.daily_summary(today)
        current_hour = datetime.now().hour

        return render_template(
            "index.html",
            date=today,
            earliest_date=earliest or today,
            recent_detections=recent,
            daily_summary=summary,
            current_hour=current_hour,
        )

    @app.route("/daily_summary/<date>")
    def daily_summary(date):
        db = current_app.config["db"]
        earliest = db.earliest_detection_date()
        summary = db.daily_summary(date)

        return render_template(
            "daily_summary.html",
            date=date,
            earliest_date=earliest or date,
            daily_summary=summary,
        )

    @app.route("/detections/by_hour/<date>/<int:hour>")
    def detections_by_hour(date, hour):
        db = current_app.config["db"]
        detections = db.detections_by_hour(date, hour)

        return render_template(
            "detections_by_hour.html",
            date=date,
            hour=hour,
            detections=detections,
        )

    @app.route("/detections/by_species/<path:scientific_name>/<date>")
    def detections_by_species(scientific_name, date):
        db = current_app.config["db"]
        detections = db.detections_by_species(scientific_name, date)
        common_name = detections[0]["common_name"] if detections else scientific_name

        return render_template(
            "detections_by_species.html",
            date=date,
            scientific_name=scientific_name,
            common_name=common_name,
            detections=detections,
        )

    @app.route("/media/<path:filename>")
    def serve_media(filename):
        storage = current_app.config["storage"]
        return send_from_directory(storage.base_dir, filename)

    @app.route("/api/status")
    def api_status():
        db = current_app.config["db"]
        today = datetime.now().strftime("%Y-%m-%d")
        summary = db.daily_summary(today)
        total_today = sum(s["total"] for s in summary.values())

        return jsonify({
            "status": "running",
            "events_today": total_today,
            "species_today": len(summary),
        })

    @app.route("/live")
    def live():
        return render_template("live.html")

    @app.route("/api/stream_status")
    def stream_status():
        consumer = current_app.config["consumer"]
        pipeline = current_app.config["pipeline"]
        config = current_app.config["app_config"]
        return jsonify({
            "frigate_connected": consumer.connected,
            "cameras": config.frigate.cameras,
            "events_today": pipeline.events_today,
            "last_detection": pipeline.last_detection_info,
        })

    @app.route("/api/frigate_stream_urls")
    def frigate_stream_urls():
        import requests as req
        config = current_app.config["app_config"]
        fc = config.frigate
        cameras = fc.cameras
        # If no cameras configured, fetch list from Frigate
        if not cameras:
            try:
                base = f"http://{fc.host}:{fc.port}"
                resp = req.get(f"{base}/api/config", timeout=5)
                resp.raise_for_status()
                frigate_config = resp.json()
                cameras = list(frigate_config.get("cameras", {}).keys())
            except Exception:
                cameras = []
        return jsonify({
            "cameras": cameras,
        })

    @app.route("/api/frigate/snapshot/<camera>")
    def frigate_snapshot_proxy(camera):
        import requests as req
        from flask import Response
        config = current_app.config["app_config"]
        fc = config.frigate
        url = f"http://{fc.host}:{fc.port}/api/{camera}/latest.jpg"
        try:
            resp = req.get(url, timeout=fc.api_timeout)
            resp.raise_for_status()
            return Response(resp.content, mimetype="image/jpeg")
        except Exception:
            return Response(status=502)

    @app.route("/api/frigate/clip/<event_id>")
    def frigate_clip_proxy(event_id):
        import requests as req
        from flask import Response
        logger = logging.getLogger(__name__)
        config = current_app.config["app_config"]
        fc = config.frigate
        db = current_app.config["db"]
        pre_pad = db.get_setting("clip_pre_padding", fc.clip_pre_padding)
        post_pad = db.get_setting("clip_post_padding", fc.clip_post_padding)

        # Try to get an extended clip from recordings if padding is configured
        url = f"http://{fc.host}:{fc.port}/api/events/{event_id}/clip.mp4"
        if pre_pad > 0 or post_pad > 0:
            try:
                event_resp = req.get(
                    f"http://{fc.host}:{fc.port}/api/events/{event_id}",
                    timeout=fc.api_timeout,
                )
                event_resp.raise_for_status()
                ev = event_resp.json()
                camera = ev.get("camera")
                start_ts = ev.get("start_time")
                end_ts = ev.get("end_time")
                if camera and start_ts and end_ts:
                    padded_start = float(start_ts) - pre_pad
                    padded_end = float(end_ts) + post_pad
                    url = (
                        f"http://{fc.host}:{fc.port}/api/{camera}"
                        f"/start/{padded_start}/end/{padded_end}/clip.mp4"
                    )
            except Exception:
                logger.debug("Could not fetch event details for padding, using default clip")

        try:
            # Fetch clip from Frigate
            logger.info("Fetching clip from Frigate: %s", url)
            resp = req.get(url, timeout=max(fc.api_timeout, 30))
            resp.raise_for_status()
            data = resp.content
            total = len(data)
            logger.info("Clip fetched: %d bytes, Range: %s", total, request.headers.get("Range", "none"))

            # Handle Range requests (required for Safari/iOS video playback)
            range_header = request.headers.get("Range")
            if range_header:
                # Parse "bytes=start-end"
                byte_range = range_header.strip().split("=")[1]
                parts = byte_range.split("-")
                start = int(parts[0])
                end = int(parts[1]) if parts[1] else total - 1
                end = min(end, total - 1)
                length = end - start + 1

                return Response(
                    data[start:end + 1],
                    status=206,
                    mimetype="video/mp4",
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{total}",
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(length),
                        "Content-Disposition": f"inline; filename={event_id}_clip.mp4",
                    },
                )

            return Response(
                data,
                status=200,
                mimetype="video/mp4",
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(total),
                    "Content-Disposition": f"inline; filename={event_id}_clip.mp4",
                },
            )
        except Exception:
            return Response(status=502)

    @app.route("/logs")
    def logs():
        return render_template("logs.html")

    @app.route("/api/logs")
    def api_logs():
        limit = request.args.get("limit", 100, type=int)
        logs = log_buffer.get_logs(limit=min(limit, 500))
        return jsonify({"logs": logs})

    @app.route("/api/logs/filter_status")
    def api_logs_filter_status():
        werkzeug_filter = current_app.config.get("werkzeug_filter")
        werkzeug_logger = logging.getLogger("werkzeug")
        is_active = werkzeug_filter in werkzeug_logger.filters if werkzeug_filter else False
        return jsonify({"filter_internal_ips": is_active})

    @app.route("/api/logs/filter_toggle", methods=["POST"])
    def api_logs_filter_toggle():
        werkzeug_filter = current_app.config.get("werkzeug_filter")
        if not werkzeug_filter:
            return jsonify({"error": "Filter not available"}), 400

        werkzeug_logger = logging.getLogger("werkzeug")
        is_active = werkzeug_filter in werkzeug_logger.filters

        if is_active:
            werkzeug_logger.removeFilter(werkzeug_filter)
        else:
            werkzeug_logger.addFilter(werkzeug_filter)

        return jsonify({"filter_internal_ips": not is_active})

    @app.route("/settings")
    def settings():
        return render_template("settings.html")

    @app.route("/api/settings")
    def api_settings():
        db = current_app.config["db"]
        app_config = current_app.config["app_config"]

        classification_threshold = db.get_setting("classification_threshold", app_config.classification.threshold)
        clip_pre_padding = db.get_setting("clip_pre_padding", app_config.frigate.clip_pre_padding)
        clip_post_padding = db.get_setting("clip_post_padding", app_config.frigate.clip_post_padding)

        return jsonify({
            "classification_threshold": classification_threshold,
            "clip_pre_padding": clip_pre_padding,
            "clip_post_padding": clip_post_padding,
        })

    @app.route("/api/settings", methods=["POST"])
    def api_settings_update():
        db = current_app.config["db"]
        data = request.get_json()

        if "classification_threshold" in data:
            value = float(data["classification_threshold"])
            if 0 <= value <= 1:
                db.set_setting("classification_threshold", value)
            else:
                return jsonify({"error": "classification_threshold must be between 0 and 1"}), 400

        if "clip_pre_padding" in data:
            value = int(data["clip_pre_padding"])
            if 0 <= value <= 120:
                db.set_setting("clip_pre_padding", value)
            else:
                return jsonify({"error": "clip_pre_padding must be between 0 and 120"}), 400

        if "clip_post_padding" in data:
            value = int(data["clip_post_padding"])
            if 0 <= value <= 120:
                db.set_setting("clip_post_padding", value)
            else:
                return jsonify({"error": "clip_post_padding must be between 0 and 120"}), 400

        return jsonify({"success": True})
