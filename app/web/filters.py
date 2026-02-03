from datetime import datetime

from flask import Flask


def register_filters(app: Flask):
    @app.template_filter("datetime")
    def format_datetime(value, fmt="%Y-%m-%d %H:%M:%S"):
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except (ValueError, TypeError):
                return value
        if isinstance(value, datetime):
            return value.strftime(fmt)
        return value

    @app.template_filter("time_only")
    def format_time(value):
        return format_datetime(value, "%H:%M:%S")

    @app.template_filter("pct")
    def format_percent(value):
        try:
            return f"{float(value) * 100:.1f}%"
        except (ValueError, TypeError):
            return value
