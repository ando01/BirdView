import logging

from flask import Flask

from app.config import AppConfig
from app.db import Database
from app.frigate import FrigateConsumer
from app.pipeline import FrigatePipeline
from app.storage import ImageStorage
from app.web.filters import register_filters
from app.web.routes import register_routes


def create_app(
    db: Database,
    storage: ImageStorage,
    config: AppConfig,
    consumer: FrigateConsumer,
    pipeline: FrigatePipeline,
    werkzeug_filter: logging.Filter = None,
) -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    app.config["db"] = db
    app.config["storage"] = storage
    app.config["app_config"] = config
    app.config["consumer"] = consumer
    app.config["pipeline"] = pipeline
    app.config["werkzeug_filter"] = werkzeug_filter

    register_filters(app)
    register_routes(app)

    return app
