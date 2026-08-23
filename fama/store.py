"""SQLite persistence: events, tasks, evidence, decisions, memory, metrics.

Schema-light by design: rows carry JSON payloads so the subsystems can
evolve without migrations.  Everything the World UI shows is served from
this real state (sec. 38).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
  scope TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
  PRIMARY KEY (scope, key)
);
CREATE INDEX IF NOT EXISTS kv_scope ON kv(scope);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, ts TEXT NOT NULL, task_id TEXT, type TEXT NOT NULL,
  level TEXT NOT NULL, phase TEXT, title TEXT, payload TEXT
);
CREATE INDEX IF NOT EXISTS events_task ON events(task_id);
"""


class Store:
    def __init__(self, path: str | None = None):
        if path is None:
            path = os.environ.get("FAMA_DB", str(Path(os.environ.get("FAMA_HOME", ".fama")) / "fama.db"))
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self):
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------ kv helpers

    def put(self, scope: str, key: str, value: Any):
        with self._lock:
            self._db.execute(
                "INSERT INTO kv(scope,key,value) VALUES(?,?,?) "
                "ON CONFLICT(scope,key) DO UPDATE SET value=excluded.value",
                (scope, key, json.dumps(value, ensure_ascii=False, default=str)))
            self._db.commit()

    def get(self, scope: str, key: str) -> Optional[Any]:
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE scope=? AND key=?",
                                   (scope, key)).fetchone()
        return json.loads(row[0]) if row else None

    def delete(self, scope: str, key: str):
        with self._lock:
            self._db.execute("DELETE FROM kv WHERE scope=? AND key=?", (scope, key))
            self._db.commit()

    def list(self, scope: str, prefix: str = "") -> list[tuple[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT key,value FROM kv WHERE scope=? AND key LIKE ? ORDER BY key",
                (scope, prefix + "%")).fetchall()
        return [(k, json.loads(v)) for k, v in rows]

    # ------------------------------------------------------------ events

    def add_event(self, ev: dict):
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO events(id,ts,task_id,type,level,phase,title,payload) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (ev["id"], ev["ts"], ev.get("task_id"), ev["type"], ev.get("level", "info"),
                 ev.get("phase", ""), ev.get("title", ""),
                 json.dumps(ev.get("payload", {}), ensure_ascii=False, default=str)))
            self._db.commit()

    def events(self, task_id: Optional[str] = None, limit: int = 2000,
               offset: int = 0) -> list[dict]:
        q = "SELECT id,ts,task_id,type,level,phase,title,payload FROM events"
        args: list = []
        if task_id:
            q += " WHERE task_id=?"
            args.append(task_id)
        q += " ORDER BY rowid DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        out = []
        for r in reversed(rows):
            out.append({"id": r[0], "ts": r[1], "task_id": r[2], "type": r[3],
                        "level": r[4], "phase": r[5], "title": r[6],
                        "payload": json.loads(r[7] or "{}")})
        return out

    # ------------------------------------------------------------ misc

    def stats(self) -> dict:
        with self._lock:
            n_ev = self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            n_kv = self._db.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
        return {"events": n_ev, "kv_rows": n_kv, "path": self.path}
