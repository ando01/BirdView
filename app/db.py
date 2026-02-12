import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                detection_time TIMESTAMP NOT NULL,
                duration_seconds REAL DEFAULT 0,
                score REAL NOT NULL,
                scientific_name TEXT NOT NULL,
                common_name TEXT NOT NULL,
                detection_confidence REAL NOT NULL,
                snapshot_path TEXT,
                thumbnail_path TEXT,
                clip_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_detections_time
                ON detections(detection_time);
            CREATE INDEX IF NOT EXISTS idx_detections_species
                ON detections(scientific_name);
            CREATE INDEX IF NOT EXISTS idx_detections_date
                ON detections(date(detection_time));

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        conn.close()
        logger.info("Database initialized at %s", self._db_path)

    def insert_detection(
        self,
        event_id: str,
        detection_time: datetime,
        duration_seconds: float,
        score: float,
        scientific_name: str,
        common_name: str,
        detection_confidence: float,
        snapshot_path: Optional[str] = None,
        thumbnail_path: Optional[str] = None,
        clip_path: Optional[str] = None,
    ):
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO detections
                (event_id, detection_time, duration_seconds, score,
                 scientific_name, common_name, detection_confidence,
                 snapshot_path, thumbnail_path, clip_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    detection_time.isoformat(),
                    duration_seconds,
                    score,
                    scientific_name,
                    common_name,
                    detection_confidence,
                    snapshot_path,
                    thumbnail_path,
                    clip_path,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def update_clip_path(self, event_id: str, clip_path: str):
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE detections SET clip_path = ? WHERE event_id = ?",
                (clip_path, event_id),
            )
            conn.commit()
        finally:
            conn.close()

    def recent_detections(self, limit: int = 10) -> List[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM detections ORDER BY detection_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def daily_summary(self, date_str: str) -> dict:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT scientific_name, common_name,
                       CAST(strftime('%H', detection_time) AS INTEGER) AS hour,
                       COUNT(*) AS count
                FROM detections
                WHERE date(detection_time) = ?
                GROUP BY scientific_name, hour
                ORDER BY count DESC
                """,
                (date_str,),
            ).fetchall()

            summary = {}
            for row in rows:
                name = row["scientific_name"]
                if name not in summary:
                    summary[name] = {
                        "scientific_name": name,
                        "common_name": row["common_name"],
                        "total": 0,
                        "hourly": [0] * 24,
                    }
                summary[name]["hourly"][row["hour"]] = row["count"]
                summary[name]["total"] += row["count"]

            return dict(
                sorted(summary.items(), key=lambda x: x[1]["total"], reverse=True)
            )
        finally:
            conn.close()

    def detections_by_hour(self, date_str: str, hour: int) -> List[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM detections
                WHERE date(detection_time) = ?
                  AND CAST(strftime('%H', detection_time) AS INTEGER) = ?
                ORDER BY detection_time
                """,
                (date_str, hour),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def detections_by_species(
        self, scientific_name: str, date_str: str
    ) -> List[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM detections
                WHERE scientific_name = ? AND date(detection_time) = ?
                ORDER BY detection_time
                """,
                (scientific_name, date_str),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def earliest_detection_date(self) -> Optional[str]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT MIN(date(detection_time)) AS d FROM detections"
            ).fetchone()
            return row["d"] if row and row["d"] else None
        finally:
            conn.close()

    def cleanup_old_detections(self, retention_days: int) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                DELETE FROM detections
                WHERE date(detection_time) < date('now', ? || ' days')
                """,
                (f"-{retention_days}",),
            )
            count = cursor.rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def get_setting(self, key: str, default=None):
        """Get a single setting value. Returns default if not found."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
            if row:
                try:
                    return json.loads(row["value"])
                except (json.JSONDecodeError, TypeError):
                    return row["value"]
            return default
        finally:
            conn.close()

    def get_all_settings(self) -> dict:
        """Get all settings as a dictionary."""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
            settings = {}
            for row in rows:
                try:
                    settings[row["key"]] = json.loads(row["value"])
                except (json.JSONDecodeError, TypeError):
                    settings[row["key"]] = row["value"]
            return settings
        finally:
            conn.close()

    def set_setting(self, key: str, value):
        """Set a setting value. Value will be JSON-encoded if not a string."""
        conn = self._get_conn()
        try:
            if not isinstance(value, str):
                value = json.dumps(value)
            conn.execute(
                """
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (key, value),
            )
            conn.commit()
        finally:
            conn.close()

    def update_settings(self, settings: dict):
        """Update multiple settings at once."""
        for key, value in settings.items():
            self.set_setting(key, value)
