"""SQLite persistence and deterministic starter data for PROMETHEUS-X."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("PROMETHEUS_DB", Path(__file__).parent / "data" / "prometheus.db"))


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.row_factory = sqlite3.Row
    return db


def initialize() -> None:
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS batches (
          id TEXT PRIMARY KEY, product TEXT NOT NULL, units INTEGER NOT NULL,
          inspected INTEGER NOT NULL, passed INTEGER NOT NULL, micro_scratch_sla REAL NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS inspection_logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT, serial_id TEXT, batch_id TEXT,
          verdict TEXT NOT NULL, defect_type TEXT NOT NULL, severity REAL NOT NULL,
          confidence REAL NOT NULL, latency_ms REAL NOT NULL, result_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS serials (
          serial_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, product TEXT NOT NULL,
          verdict TEXT NOT NULL, defect_type TEXT NOT NULL, severity REAL NOT NULL,
          confidence REAL NOT NULL, heatmap TEXT, issued_at TEXT NOT NULL,
          FOREIGN KEY(batch_id) REFERENCES batches(id)
        );
        CREATE INDEX IF NOT EXISTS idx_logs_created ON inspection_logs(created_at);
        """)
        if db.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 0:
            now = datetime.now(timezone.utc)
            batches = [
                ("BATCH-2401", "Precision metal nut", 1200, 1200, 1187, 99.2, now - timedelta(days=1)),
                ("BATCH-2402", "Precision metal nut", 860, 860, 843, 98.8, now - timedelta(days=2)),
                ("BATCH-2403", "Machined valve housing", 640, 640, 627, 99.0, now - timedelta(days=3)),
            ]
            db.executemany("INSERT INTO batches VALUES(?,?,?,?,?,?,?)", [(*b[:6], b[6].isoformat()) for b in batches])
            samples = [
                ("PX-2026-000184", "BATCH-2401"),
                ("PX-2026-000185", "BATCH-2401"),
                ("PX-2026-000276", "BATCH-2402"),
            ]
            for serial, batch in samples:
                db.execute("INSERT INTO serials VALUES(?,?,?,?,?,?,?,?,?)", (
                    serial, batch, "Precision metal nut", "PASS", "No defect detected", 0.0,
                    98.4, None, now.isoformat(),
                ))


def add_inspection(result: dict[str, Any], serial_id: str | None = None, batch_id: str | None = None) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        cur = db.execute("""INSERT INTO inspection_logs
          (serial_id,batch_id,verdict,defect_type,severity,confidence,latency_ms,result_json,created_at)
          VALUES(?,?,?,?,?,?,?,?,?)""", (
            serial_id, batch_id, result["verdict"], result["defect_type"], result["severity"],
            result["confidence"], result["latency_ms"], json.dumps(result), now,
        ))
        if serial_id:
            db.execute("""INSERT INTO serials VALUES(?,?,?,?,?,?,?,?,?)
              ON CONFLICT(serial_id) DO UPDATE SET verdict=excluded.verdict,
              defect_type=excluded.defect_type,severity=excluded.severity,
              confidence=excluded.confidence,heatmap=excluded.heatmap,issued_at=excluded.issued_at""", (
                serial_id, batch_id or "BATCH-LIVE", "Inspected product", result["verdict"],
                result["defect_type"], result["severity"], result["confidence"],
                result.get("heatmap"), now,
            ))
        return int(cur.lastrowid)


def list_batches() -> list[dict[str, Any]]:
    with connect() as db:
        return [dict(row) for row in db.execute("SELECT * FROM batches ORDER BY created_at DESC")]


def recent_inspections(limit: int = 20) -> list[dict[str, Any]]:
    with connect() as db:
        return [dict(row) for row in db.execute(
            "SELECT id,serial_id,batch_id,verdict,defect_type,severity,confidence,latency_ms,created_at "
            "FROM inspection_logs ORDER BY id DESC LIMIT ?", (limit,)
        )]


def certificate(serial_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM serials WHERE serial_id=?", (serial_id,)).fetchone()
        return dict(row) if row else None


def seed_serials() -> list[str]:
    with connect() as db:
        return [r[0] for r in db.execute("SELECT serial_id FROM serials ORDER BY serial_id")]
