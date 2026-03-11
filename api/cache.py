import sqlite3
import json
import time
import threading
from pathlib import Path
from typing import Any, Optional


class CacheManager:
    """SQLite-backed TTL cache. Thread-safe via write lock + WAL journal mode."""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # WAL mode: concurrent readers don't block writes, safe for multi-threaded Streamlit
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL
                )
            """)
            self.conn.commit()

    def get(self, key: str) -> Optional[Any]:
        # Reads are safe without lock in WAL mode
        row = self.conn.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        value, expires_at = row
        if expires_at != -1 and time.time() > expires_at:
            self.delete(key)
            return None
        return json.loads(value)

    def set(self, key: str, value: Any, ttl: int = 3600):
        expires_at = -1 if ttl == -1 else time.time() + ttl
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), expires_at),
            )
            self.conn.commit()

    def delete(self, key: str):
        with self._lock:
            self.conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self.conn.commit()

    def purge_expired(self):
        with self._lock:
            self.conn.execute(
                "DELETE FROM cache WHERE expires_at != -1 AND expires_at < ?", (time.time(),)
            )
            self.conn.commit()
