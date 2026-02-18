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
        config = current_app.config["app_config"]
        fc = config.frigate
        base = f"http://{fc.host}:{fc.port}"
        return jsonify({
            "cameras": fc.cameras,
            "mjpeg_base": f"{base}/api",
            "latest_base": f"{base}/api",
        })

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

        return jsonify({"classification_threshold": classification_threshold})

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

        return jsonify({"success": True})
