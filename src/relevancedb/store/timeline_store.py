"""
timeline_store.py — temporal storage head.

Uses SQLite (stdlib, zero deps).
Tracks every document version with timestamps and decay weights.

This solves the problem where a policy from 2021 and its 2024
replacement score equally in a vector search. Here, recency matters.

Key concepts:
  - Every ingest of a doc creates a new version row (append-only).
  - decay_weight: float 0-1, computed from age. Recent = 1.0, old = lower.
  - "as of" queries: return the version that existed at a given date.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class DocVersion:
    doc_path: str
    version: int
    ingested_at: str    
    char_count: int
    decay_weight: float  # 1.0 = fresh, approaches 0 as doc ages
    format: str


class TimelineStore:
    """
    Wraps SQLite. Append-only document version history.

    Args:
        data_dir: Directory where the SQLite file is stored.
        decay_days: Half-life in days for decay weight.
                    Default 180 — a doc loses half its weight every 6 months.
    """

    def __init__(self, data_dir: Path, decay_days: int = 180) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.decay_days = decay_days
        self._db_path = self.data_dir / "timeline.db"
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()


    def add(self, doc_path: str, char_count: int, fmt: str) -> DocVersion:
        """
        Record a new version of a document.

        Every call creates a new row — we never update existing rows.
        This gives us full version history for free.

        Returns the DocVersion that was just inserted.
        """
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        cur = conn.execute(
            "SELECT COUNT(*) FROM doc_versions WHERE doc_path = ?",
            (doc_path,),
        )
        version = cur.fetchone()[0] + 1

        decay = self._compute_decay(now)

        conn.execute(
            """
            INSERT INTO doc_versions
                (doc_path, version, ingested_at, char_count, decay_weight, format)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (doc_path, version, now, char_count, decay, fmt),
        )
        conn.commit()

        return DocVersion(
            doc_path=doc_path,
            version=version,
            ingested_at=now,
            char_count=char_count,
            decay_weight=decay,
            format=fmt,
        )

    def latest(self, doc_path: str) -> DocVersion | None:
        """Return the most recent version of a document."""
        conn = self._get_conn()
        cur = conn.execute(
            """
            SELECT doc_path, version, ingested_at, char_count, decay_weight, format
            FROM doc_versions
            WHERE doc_path = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (doc_path,),
        )
        row = cur.fetchone()
        return DocVersion(*row) if row else None

    def as_of(self, doc_path: str, date: str) -> DocVersion | None:
        """
        Return the version that existed at a given ISO date string.

        Useful for point-in-time queries:
            db.query("what was the policy?", as_of="2023-06-01")
        """
        conn = self._get_conn()
        cur = conn.execute(
            """
            SELECT doc_path, version, ingested_at, char_count, decay_weight, format
            FROM doc_versions
            WHERE doc_path = ? AND ingested_at <= ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (doc_path, date),
        )
        row = cur.fetchone()
        return DocVersion(*row) if row else None

    def all_docs(self) -> list[str]:
        """Return unique doc paths that have been ingested."""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT DISTINCT doc_path FROM doc_versions ORDER BY doc_path"
        )
        return [row[0] for row in cur.fetchall()]

    def decay_weight(self, doc_path: str) -> float:
        """
        Return current decay weight for the latest version of a doc.
        Used by the fusion ranker to down-rank stale content.
        """
        v = self.latest(doc_path)
        if v is None:
            return 1.0
        return self._compute_decay(v.ingested_at)

    def refresh_decay(self) -> None:
        """
        Recompute decay weights for all docs based on current time.
        Call this periodically (e.g. daily) to keep weights fresh.
        """
        conn = self._get_conn()
        cur = conn.execute("SELECT rowid, ingested_at FROM doc_versions")
        rows = cur.fetchall()
        for rowid, ingested_at in rows:
            new_weight = self._compute_decay(ingested_at)
            conn.execute(
                "UPDATE doc_versions SET decay_weight = ? WHERE rowid = ?",
                (new_weight, rowid),
            )
        conn.commit()


    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_versions (
                doc_path     TEXT NOT NULL,
                version      INTEGER NOT NULL,
                ingested_at  TEXT NOT NULL,
                char_count   INTEGER NOT NULL,
                decay_weight REAL NOT NULL DEFAULT 1.0,
                format       TEXT NOT NULL DEFAULT 'txt'
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_doc_path ON doc_versions (doc_path)"
        )
        conn.commit()

    def _compute_decay(self, ingested_at: str) -> float:
        """
        Exponential decay: weight = 0.5 ^ (age_days / decay_days)

        Examples with decay_days=180:
          0 days old  → 1.0
          180 days    → 0.5
          360 days    → 0.25
          720 days    → 0.125
        """
        try:
            ingested = datetime.fromisoformat(ingested_at)
            if ingested.tzinfo is None:
                ingested = ingested.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            age_days = (now - ingested).total_seconds() / 86400
            return round(math.pow(0.5, age_days / self.decay_days), 6)
        except Exception:
            return 1.0