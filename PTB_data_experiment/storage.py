"""One transactional database, compressed arrays, no per-candidate parameter dumps."""
from __future__ import annotations
import hashlib
import io
import json
import sqlite3
from pathlib import Path
import numpy as np


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pack_arrays(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def unpack_arrays(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as file:
        return {k: file[k].copy() for k in file.files}


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY, specification TEXT NOT NULL, status TEXT NOT NULL,
            metadata TEXT, selection TEXT, error TEXT, updated REAL);
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, target_seed INTEGER,
            rep_seed INTEGER, method TEXT, lr REAL, status TEXT,
            auc REAL, final_loss REAL, metadata TEXT, trace BLOB NOT NULL);
        CREATE INDEX IF NOT EXISTS runs_case ON runs(case_id);
        CREATE TABLE IF NOT EXISTS diagnostics (
            run_id TEXT PRIMARY KEY, case_id TEXT, metadata TEXT, arrays BLOB);
        CREATE TABLE IF NOT EXISTS validation (name TEXT PRIMARY KEY, result TEXT);
        """)
        self.connection.commit()
        # run_id already includes the numerical protocol. The old LR-only
        # uniqueness constraint replaced a prior protocol during a correction.
        schema = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='runs'").fetchone()[0]
        if "UNIQUE(case_id,target_seed,rep_seed,method,lr)" in schema.replace(" ", "").replace("\n", ""):
            with self.connection:
                self.connection.execute("""CREATE TABLE runs_protocol_safe (
                    run_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, target_seed INTEGER,
                    rep_seed INTEGER, method TEXT, lr REAL, status TEXT,
                    auc REAL, final_loss REAL, metadata TEXT, trace BLOB NOT NULL)""")
                self.connection.execute("INSERT INTO runs_protocol_safe SELECT * FROM runs")
                before = self.connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
                after = self.connection.execute("SELECT COUNT(*) FROM runs_protocol_safe").fetchone()[0]
                if before != after:
                    raise RuntimeError("Protocol-safe storage migration lost records")
                self.connection.execute("DROP TABLE runs")
                self.connection.execute("ALTER TABLE runs_protocol_safe RENAME TO runs")
                self.connection.execute("CREATE INDEX runs_case ON runs(case_id)")

    def save_run(self, run_id, case_id, target_seed, rep_seed, method, lr, metadata, arrays):
        self.connection.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, case_id, target_seed, rep_seed, method, lr, metadata["status"],
             metadata.get("auc"), metadata.get("final_loss"),
             json.dumps(metadata, allow_nan=True), pack_arrays(arrays)))
        self.connection.commit()

    def run(self, run_id):
        row = self.connection.execute("SELECT metadata,trace FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return None if row is None else (json.loads(row[0]), unpack_arrays(row[1]))

    def summaries(self, case_id):
        return [json.loads(row[0]) for row in self.connection.execute(
            "SELECT metadata FROM runs WHERE case_id=?", (case_id,))]

    def save_diagnostics(self, run_id, case_id, metadata, arrays):
        self.connection.execute("INSERT OR REPLACE INTO diagnostics VALUES (?,?,?,?)",
                                (run_id, case_id, json.dumps(metadata, allow_nan=True), pack_arrays(arrays)))
        self.connection.commit()

    def has_diagnostics(self, run_id, version=1):
        row=self.connection.execute("SELECT metadata FROM diagnostics WHERE run_id=?", (run_id,)).fetchone()
        return row is not None and json.loads(row[0]).get("diagnostics_version",1)>=version

    def close(self):
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()
