import logging
from collections import deque
from datetime import datetime
from threading import Lock


class LogBuffer(logging.Handler):
    """A logging handler that stores recent log entries in memory."""

    def __init__(self, max_entries=500):
        super().__init__()
        self._buffer = deque(maxlen=max_entries)
        self._lock = Lock()

    def emit(self, record):
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(record.created).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            }
            with self._lock:
                self._buffer.append(entry)
        except Exception:
            self.handleError(record)

    def get_logs(self, limit=100):
        """Return recent log entries, newest first."""
        with self._lock:
            logs = list(self._buffer)
        return list(reversed(logs[-limit:]))

    def clear(self):
        """Clear the log buffer."""
        with self._lock:
            self._buffer.clear()


# Global log buffer instance
log_buffer = LogBuffer(max_entries=500)
