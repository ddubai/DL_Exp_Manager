"""SQLite 데이터 계층.

로컬 디렉토리에 `experiments.db` 를 만들고 Task -> Work -> Run 계층을 관리한다.

    tasks (Level 1: DL Task)
      └─ works (Level 2: Work ID)
           ├─ train_runs
           └─ inference_runs
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sqlite3
from datetime import datetime
from typing import Any, Iterable, Sequence

from . import constants as C
from .utils import dumps_metrics, now_iso

SCHEMA_VERSION = 7

DEFAULT_DB_NAME = "experiments.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS servers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    host        TEXT    DEFAULT '',
    gpu         TEXT    DEFAULT '',
    note        TEXT    DEFAULT '',
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT    DEFAULT '',
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS works (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    name        TEXT    NOT NULL,
    description TEXT    DEFAULT '',
    created_at  TEXT    NOT NULL,
    UNIQUE(task_id, name)
);

CREATE TABLE IF NOT EXISTS train_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    work_id       INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    server        TEXT    DEFAULT '',
    model         TEXT    DEFAULT '',
    dataset       TEXT    DEFAULT '',
    dataset_path  TEXT    DEFAULT '',
    result_path   TEXT    DEFAULT '',
    status        TEXT    DEFAULT 'done',
    started_at    TEXT    DEFAULT '',
    duration_sec  REAL,
    epochs        TEXT    DEFAULT '',
    batch_size    TEXT    DEFAULT '',
    crop_size     TEXT    DEFAULT '',
    lr            TEXT    DEFAULT '',
    optimizer     TEXT    DEFAULT '',
    gpu_indices   TEXT    DEFAULT '',
    extra_json    TEXT    DEFAULT '{}',
    metrics_json  TEXT    DEFAULT '{}',
    exec_command  TEXT    DEFAULT '',
    config_yaml   TEXT    DEFAULT '',
    notes         TEXT    DEFAULT '',
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS inference_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    work_id         INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    server          TEXT    DEFAULT '',
    model           TEXT    DEFAULT '',
    checkpoint_path TEXT    DEFAULT '',
    dataset         TEXT    DEFAULT '',
    dataset_path    TEXT    DEFAULT '',
    result_path     TEXT    DEFAULT '',
    device          TEXT    DEFAULT '',
    input_size      TEXT    DEFAULT '',
    latency_ms      REAL,
    throughput_fps  REAL,
    status          TEXT    DEFAULT 'done',
    started_at      TEXT    DEFAULT '',
    duration_sec    REAL,
    gpu_indices     TEXT    DEFAULT '',
    extra_json      TEXT    DEFAULT '{}',
    metrics_json    TEXT    DEFAULT '{}',
    exec_command    TEXT    DEFAULT '',
    config_yaml     TEXT    DEFAULT '',
    notes           TEXT    DEFAULT '',
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS run_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_kind    TEXT    NOT NULL,   -- 'train' | 'inference'
    run_id      INTEGER NOT NULL,
    action      TEXT    NOT NULL,   -- 'created' | 'updated' | 'duplicated'
    detail      TEXT    DEFAULT '',
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    work_id       INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    name          TEXT    NOT NULL,             -- e.g. "DIV2K"
    variant       TEXT    DEFAULT '',           -- e.g. "Full Pair" / "Subset A" (blank = default)
    path          TEXT    DEFAULT '',           -- registered location
    sample_count  INTEGER,                      -- 총 데이터 개수 (모르면 NULL)
    image_size    TEXT    DEFAULT '',           -- e.g. "256x256"
    extension     TEXT    DEFAULT '',           -- e.g. "png", "tiff"
    notes         TEXT    DEFAULT '',
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL,
    UNIQUE(work_id, name, variant)
);

CREATE INDEX IF NOT EXISTS idx_works_task       ON works(task_id);
CREATE INDEX IF NOT EXISTS idx_train_work       ON train_runs(work_id);
CREATE INDEX IF NOT EXISTS idx_train_status     ON train_runs(status);
CREATE INDEX IF NOT EXISTS idx_infer_work       ON inference_runs(work_id);
CREATE INDEX IF NOT EXISTS idx_history_run      ON run_history(run_kind, run_id);
CREATE INDEX IF NOT EXISTS idx_datasets_work    ON datasets(work_id);
"""

TRAIN_FIELDS: tuple[str, ...] = (
    "work_id", "server", "model", "dataset", "dataset_path", "result_path",
    "status", "started_at", "duration_sec", "epochs", "batch_size", "crop_size", "lr",
    "optimizer", "gpu_indices", "extra_json", "metrics_json", "exec_command",
    "config_yaml", "notes", "favorite", "tags", "failure_reason",
)

INFER_FIELDS: tuple[str, ...] = (
    "work_id", "server", "model", "checkpoint_path", "dataset", "dataset_path",
    "result_path", "device", "input_size", "latency_ms", "throughput_fps",
    "status", "started_at", "duration_sec", "gpu_indices", "extra_json",
    "metrics_json", "exec_command", "config_yaml", "notes",
    "favorite", "tags", "failure_reason",
    "source_train_run_id", "checkpoint_epoch",
)

# v1 -> v2 에서 추가된 컬럼. (테이블, 컬럼, 정의)
_V2_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("train_runs", "gpu_indices", "TEXT DEFAULT ''"),
    ("train_runs", "extra_json", "TEXT DEFAULT '{}'"),
    ("inference_runs", "gpu_indices", "TEXT DEFAULT ''"),
    ("inference_runs", "extra_json", "TEXT DEFAULT '{}'"),
)

# v2 -> v3: 즐겨찾기 / 태그 / 실패 사유
_V3_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("train_runs", "favorite", "INTEGER DEFAULT 0"),
    ("train_runs", "tags", "TEXT DEFAULT ''"),
    ("train_runs", "failure_reason", "TEXT DEFAULT ''"),
    ("inference_runs", "favorite", "INTEGER DEFAULT 0"),
    ("inference_runs", "tags", "TEXT DEFAULT ''"),
    ("inference_runs", "failure_reason", "TEXT DEFAULT ''"),
)

# v3 -> v4: Inference 가 어느 Train 실행의 체크포인트를 쓰는지 + 몇 epoch/iter 인지
_V4_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("inference_runs", "source_train_run_id", "INTEGER"),
    ("inference_runs", "checkpoint_epoch", "TEXT DEFAULT ''"),
)

# v4 -> v5: 데이터셋 레지스트리에 총 데이터 개수
_V5_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("datasets", "sample_count", "INTEGER"),
)

# v5 -> v6: 데이터셋 레지스트리에 이미지 크기 / 확장자
_V6_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("datasets", "image_size", "TEXT DEFAULT ''"),
    ("datasets", "extension", "TEXT DEFAULT ''"),
)

# v6 -> v7: Train 하이퍼파라미터에 crop size
_V7_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("train_runs", "crop_size", "TEXT DEFAULT ''"),
)


def default_db_path() -> str:
    """기본 DB 경로: 프로젝트 루트의 experiments.db"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), DEFAULT_DB_NAME)


class Database:
    """실험 메타데이터 저장소 (SQLite)."""

    def __init__(self, path: str | None = None, seed: bool = True) -> None:
        self.path = os.path.abspath(path or default_db_path())
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        is_new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0

        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._create_schema()
        if is_new and seed:
            self.seed_defaults()

    # -- 내부 --------------------------------------------------------------
    def _create_schema(self) -> None:
        with self.conn:
            self.conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """기존 DB 를 현재 스키마로 올린다. 몇 번 실행해도 안전하다."""
        current = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
        with self.conn:
            for table, column, definition in (
                _V2_COLUMNS + _V3_COLUMNS + _V4_COLUMNS + _V5_COLUMNS + _V6_COLUMNS + _V7_COLUMNS
            ):
                if not self._has_column(table, column):
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.migrated_from = current if current < SCHEMA_VERSION else None

    def _has_column(self, table: str, column: str) -> bool:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(row[1] == column for row in rows)

    def _exec(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self.conn:
            return self.conn.execute(sql, tuple(params))

    @staticmethod
    def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass

    # -- 초기 데이터 --------------------------------------------------------
    def seed_defaults(self) -> None:
        ts = now_iso()
        with self.conn:
            for name, host, gpu in C.DEFAULT_SERVERS:
                self.conn.execute(
                    "INSERT OR IGNORE INTO servers(name, host, gpu, note, updated_at) "
                    "VALUES (?,?,?,'',?)",
                    (name, host, gpu, ts),
                )
            for name, desc in C.DEFAULT_TASKS:
                self.conn.execute(
                    "INSERT OR IGNORE INTO tasks(name, description, created_at) VALUES (?,?,?)",
                    (name, desc, ts),
                )
        for task_name, works in C.DEFAULT_WORKS.items():
            task = self.get_task_by_name(task_name)
            if task is None:
                continue
            for work_name, desc in works:
                self.add_work(task["id"], work_name, desc)

    # -- 서버 ---------------------------------------------------------------
    def list_servers(self) -> list[dict[str, Any]]:
        return self._rows(self.conn.execute("SELECT * FROM servers ORDER BY name"))

    def server_names(self) -> list[str]:
        return [s["name"] for s in self.list_servers()]

    def add_server(self, name: str, host: str = "", gpu: str = "", note: str = "") -> int | None:
        name = name.strip()
        if not name:
            return None
        cur = self._exec(
            "INSERT OR IGNORE INTO servers(name, host, gpu, note, updated_at) VALUES (?,?,?,?,?)",
            (name, host, gpu, note, now_iso()),
        )
        return cur.lastrowid

    def running_by_server(self) -> dict[str, list[dict[str, Any]]]:
        """서버별 현재 running 상태의 학습 목록."""
        rows = self._rows(
            self.conn.execute(
                "SELECT t.id, t.server, t.model, t.started_at, t.gpu_indices, "
                "t.exec_command, t.result_path, "
                "w.name AS work_name, tk.name AS task_name "
                "FROM train_runs t "
                "JOIN works w ON w.id = t.work_id "
                "JOIN tasks tk ON tk.id = w.task_id "
                "WHERE t.status = ? ORDER BY t.started_at DESC",
                (C.STATUS_RUNNING,),
            )
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["server"] or "(unassigned)", []).append(row)
        return grouped

    # -- Level 1: Task ------------------------------------------------------
    def list_tasks(self) -> list[dict[str, Any]]:
        return self._rows(self.conn.execute("SELECT * FROM tasks ORDER BY name COLLATE NOCASE"))

    def get_task_by_name(self, name: str) -> dict[str, Any] | None:
        cur = self.conn.execute("SELECT * FROM tasks WHERE name = ?", (name,))
        row = cur.fetchone()
        return dict(row) if row else None

    def add_task(self, name: str, description: str = "") -> int | None:
        name = name.strip()
        if not name:
            return None
        try:
            cur = self._exec(
                "INSERT INTO tasks(name, description, created_at) VALUES (?,?,?)",
                (name, description, now_iso()),
            )
        except sqlite3.IntegrityError:
            existing = self.get_task_by_name(name)
            return existing["id"] if existing else None
        return cur.lastrowid

    def update_task(self, task_id: int, name: str, description: str = "") -> None:
        self._exec(
            "UPDATE tasks SET name = ?, description = ? WHERE id = ?",
            (name.strip(), description, task_id),
        )

    def delete_task(self, task_id: int) -> None:
        self._exec("DELETE FROM tasks WHERE id = ?", (task_id,))

    # -- Level 2: Work ------------------------------------------------------
    def list_works(self, task_id: int) -> list[dict[str, Any]]:
        return self._rows(
            self.conn.execute(
                "SELECT * FROM works WHERE task_id = ? ORDER BY name COLLATE NOCASE", (task_id,)
            )
        )

    def get_work(self, work_id: int) -> dict[str, Any] | None:
        cur = self.conn.execute(
            "SELECT w.*, t.name AS task_name FROM works w "
            "JOIN tasks t ON t.id = w.task_id WHERE w.id = ?",
            (work_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def add_work(self, task_id: int, name: str, description: str = "") -> int | None:
        name = name.strip()
        if not name:
            return None
        try:
            cur = self._exec(
                "INSERT INTO works(task_id, name, description, created_at) VALUES (?,?,?,?)",
                (task_id, name, description, now_iso()),
            )
        except sqlite3.IntegrityError:
            cur2 = self.conn.execute(
                "SELECT id FROM works WHERE task_id = ? AND name = ?", (task_id, name)
            )
            row = cur2.fetchone()
            return row["id"] if row else None
        return cur.lastrowid

    def update_work(self, work_id: int, name: str, description: str = "") -> None:
        self._exec(
            "UPDATE works SET name = ?, description = ? WHERE id = ?",
            (name.strip(), description, work_id),
        )

    def delete_work(self, work_id: int) -> None:
        self._exec("DELETE FROM works WHERE id = ?", (work_id,))

    def work_ids_for_task(self, task_id: int) -> list[int]:
        return [w["id"] for w in self.list_works(task_id)]

    def counts_for_work(self, work_id: int) -> tuple[int, int]:
        train = self.conn.execute(
            "SELECT COUNT(*) FROM train_runs WHERE work_id = ?", (work_id,)
        ).fetchone()[0]
        infer = self.conn.execute(
            "SELECT COUNT(*) FROM inference_runs WHERE work_id = ?", (work_id,)
        ).fetchone()[0]
        return int(train), int(infer)

    # -- Work 별 데이터셋 레지스트리 ------------------------------------------
    # 같은 이름이라도 variant 를 다르게 두면(예: "Full Pair" / "Subset A") 한
    # 데이터셋의 여러 판을 서로 다른 경로로 따로 등록해 둘 수 있다.
    def list_datasets(self, work_id: int) -> list[dict[str, Any]]:
        return self._rows(
            self.conn.execute(
                "SELECT * FROM datasets WHERE work_id = ? ORDER BY name COLLATE NOCASE, variant COLLATE NOCASE",
                (work_id,),
            )
        )

    def get_dataset(self, dataset_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        return dict(row) if row else None

    def add_dataset(
        self,
        work_id: int,
        name: str,
        variant: str = "",
        path: str = "",
        notes: str = "",
        sample_count: int | None = None,
        image_size: str = "",
        extension: str = "",
    ) -> int | None:
        name = name.strip()
        if not name:
            return None
        ts = now_iso()
        try:
            cur = self._exec(
                "INSERT INTO datasets(work_id, name, variant, path, sample_count, image_size, "
                "extension, notes, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    work_id, name, variant.strip(), path.strip(), sample_count,
                    image_size.strip(), extension.strip(), notes.strip(), ts, ts,
                ),
            )
        except sqlite3.IntegrityError:
            existing = self.conn.execute(
                "SELECT id FROM datasets WHERE work_id = ? AND name = ? AND variant = ?",
                (work_id, name, variant.strip()),
            ).fetchone()
            return existing["id"] if existing else None
        return int(cur.lastrowid)

    def update_dataset(
        self,
        dataset_id: int,
        name: str,
        variant: str = "",
        path: str = "",
        notes: str = "",
        sample_count: int | None = None,
        image_size: str = "",
        extension: str = "",
    ) -> None:
        self._exec(
            "UPDATE datasets SET name = ?, variant = ?, path = ?, sample_count = ?, image_size = ?, "
            "extension = ?, notes = ?, updated_at = ? WHERE id = ?",
            (
                name.strip(), variant.strip(), path.strip(), sample_count,
                image_size.strip(), extension.strip(), notes.strip(), now_iso(), dataset_id,
            ),
        )

    def delete_dataset(self, dataset_id: int) -> None:
        self._exec("DELETE FROM datasets WHERE id = ?", (dataset_id,))

    def count_runs_using_dataset(self, work_id: int, name: str, variant: str = "") -> int:
        """이름(+variant)이 일치하는 실행이 몇 건인지 - 삭제 확인창에 쓴다."""
        label = f"{name} · {variant}" if variant else name
        total = 0
        for table in ("train_runs", "inference_runs"):
            row = self.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE work_id = ? AND dataset = ?",
                (work_id, label),
            ).fetchone()
            total += int(row[0])
        return total

    # -- Level 3 데이터: Runs ----------------------------------------------
    def _select_runs(
        self,
        table: str,
        work_id: int | None = None,
        task_id: int | None = None,
    ) -> list[dict[str, Any]]:
        sql = (
            f"SELECT r.*, w.name AS work_name, t.name AS task_name, t.id AS task_id "
            f"FROM {table} r "
            f"JOIN works w ON w.id = r.work_id "
            f"JOIN tasks t ON t.id = w.task_id"
        )
        params: list[Any] = []
        if work_id is not None:
            sql += " WHERE r.work_id = ?"
            params.append(work_id)
        elif task_id is not None:
            sql += " WHERE t.id = ?"
            params.append(task_id)
        sql += " ORDER BY r.id DESC"
        return self._rows(self.conn.execute(sql, tuple(params)))

    def list_train_runs(
        self, work_id: int | None = None, task_id: int | None = None
    ) -> list[dict[str, Any]]:
        return self._select_runs("train_runs", work_id, task_id)

    def list_inference_runs(
        self, work_id: int | None = None, task_id: int | None = None
    ) -> list[dict[str, Any]]:
        return self._select_runs("inference_runs", work_id, task_id)

    def get_run(self, kind: str, run_id: int) -> dict[str, Any] | None:
        table = self._table(kind)
        cur = self.conn.execute(f"SELECT * FROM {table} WHERE id = ?", (run_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    @staticmethod
    def _table(kind: str) -> str:
        if kind == "train":
            return "train_runs"
        if kind == "inference":
            return "inference_runs"
        raise ValueError(f"Unknown run kind: {kind!r}")

    @staticmethod
    def _fields(kind: str) -> tuple[str, ...]:
        return TRAIN_FIELDS if kind == "train" else INFER_FIELDS

    def insert_run(
        self,
        kind: str,
        data: dict[str, Any],
        history_action: str = "created",
        history_detail: str = "Run created.",
    ) -> int:
        table, fields = self._table(kind), self._fields(kind)
        payload = self._normalize(data, fields)
        ts = now_iso()
        cols = list(fields) + ["created_at", "updated_at"]
        values = [payload.get(f) for f in fields] + [ts, ts]
        placeholders = ",".join("?" * len(cols))
        cur = self._exec(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})", values
        )
        run_id = int(cur.lastrowid)
        self._record_history(kind, run_id, history_action, history_detail)
        return run_id

    def update_run(self, kind: str, run_id: int, data: dict[str, Any]) -> None:
        table, fields = self._table(kind), self._fields(kind)
        old = self.get_run(kind, run_id)
        payload = self._normalize(data, fields)
        assignments = ",".join(f"{f} = ?" for f in fields)
        values = [payload.get(f) for f in fields] + [now_iso(), run_id]
        self._exec(f"UPDATE {table} SET {assignments}, updated_at = ? WHERE id = ?", values)
        if old is not None:
            detail = self._diff_summary(old, payload, fields)
            if detail:
                self._record_history(kind, run_id, "updated", detail)

    def delete_runs(self, kind: str, run_ids: Iterable[int]) -> int:
        ids = [int(i) for i in run_ids]
        if not ids:
            return 0
        table = self._table(kind)
        placeholders = ",".join("?" * len(ids))
        cur = self._exec(f"DELETE FROM {table} WHERE id IN ({placeholders})", ids)
        self._exec(
            f"DELETE FROM run_history WHERE run_kind = ? AND run_id IN ({placeholders})",
            [kind, *ids],
        )
        return cur.rowcount

    def toggle_favorite(self, kind: str, run_id: int) -> bool:
        """즐겨찾기를 뒤집고 새 상태를 돌려준다."""
        table = self._table(kind)
        row = self.conn.execute(f"SELECT favorite FROM {table} WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return False
        new_value = 0 if row["favorite"] else 1
        self._exec(
            f"UPDATE {table} SET favorite = ?, updated_at = ? WHERE id = ?",
            (new_value, now_iso(), run_id),
        )
        return bool(new_value)

    def duplicate_run(self, kind: str, run_id: int) -> int | None:
        row = self.get_run(kind, run_id)
        if row is None:
            return None
        row = dict(row)
        row["notes"] = (row.get("notes") or "").strip()
        row["notes"] = (row["notes"] + "\n(copy)").strip()
        row["status"] = C.STATUS_QUEUED
        row["favorite"] = 0
        return self.insert_run(
            kind,
            row,
            history_action="duplicated",
            history_detail=f"Duplicated from run #{run_id}.",
        )

    # -- 실행 히스토리 --------------------------------------------------------
    def _record_history(self, kind: str, run_id: int, action: str, detail: str) -> None:
        self._exec(
            "INSERT INTO run_history(run_kind, run_id, action, detail, created_at) "
            "VALUES (?,?,?,?,?)",
            (kind, run_id, action, detail, now_iso()),
        )

    def list_history(self, kind: str, run_id: int) -> list[dict[str, Any]]:
        return self._rows(
            self.conn.execute(
                "SELECT * FROM run_history WHERE run_kind = ? AND run_id = ? ORDER BY id DESC",
                (kind, run_id),
            )
        )

    # 히스토리에 사람이 읽을 만한 요약을 남길 때 무시할 필드 (JSON 은 따로 비교한다)
    _DIFF_SKIP = {"metrics_json", "extra_json"}
    _DIFF_LABELS = {"work_id": "Work"}

    def _diff_summary(
        self, old: dict[str, Any], new: dict[str, Any], fields: Sequence[str]
    ) -> str:
        changes: list[str] = []
        for field in fields:
            if field in self._DIFF_SKIP:
                continue
            old_value, new_value = old.get(field), new.get(field)
            if field == "work_id":
                if old_value == new_value:
                    continue
                old_text = self._work_label(old_value)
                new_text = self._work_label(new_value)
            else:
                old_text = "" if old_value is None else str(old_value)
                new_text = "" if new_value is None else str(new_value)
                if old_text == new_text:
                    continue
            label = self._DIFF_LABELS.get(field, field)
            changes.append(f"{label}: '{old_text or '—'}' → '{new_text or '—'}'")

        for source, label in (("metrics_json", "metric"), ("extra_json", "field")):
            old_map = self._json_map(old.get(source))
            new_map = self._json_map(new.get(source))
            for key in sorted(set(old_map) | set(new_map)):
                old_value, new_value = old_map.get(key, ""), new_map.get(key, "")
                if str(old_value) == str(new_value):
                    continue
                changes.append(
                    f"{key} ({label}): '{old_value or '—'}' → '{new_value or '—'}'"
                )
        return "; ".join(changes)

    def _work_label(self, work_id: Any) -> str:
        if not work_id:
            return ""
        work = self.get_work(int(work_id))
        return work["name"] if work else str(work_id)

    @staticmethod
    def _json_map(raw: Any) -> dict[str, Any]:
        try:
            data = json.loads(raw or "{}")
        except (TypeError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _normalize(data: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
        """폼/딕셔너리 값을 DB 컬럼 타입에 맞게 정리."""
        out: dict[str, Any] = {}
        numeric = {"duration_sec", "latency_ms", "throughput_fps"}
        for field in fields:
            value = data.get(field)
            if field == "favorite":
                out[field] = 1 if value in (True, 1, "1", "true", "True") else 0
                continue
            if field == "extra_json":
                if isinstance(value, dict):
                    value = json.dumps(
                        {str(k): v for k, v in value.items()}, ensure_ascii=False, sort_keys=True
                    )
                elif not value:
                    value = "{}"
                out[field] = value
                continue
            if field == "metrics_json":
                if isinstance(value, dict):
                    value = dumps_metrics(value)
                elif not value:
                    value = "{}"
                out[field] = value
                continue
            if field in numeric:
                if value in (None, ""):
                    out[field] = None
                else:
                    try:
                        out[field] = float(value)
                    except (TypeError, ValueError):
                        out[field] = None
                continue
            if field in ("work_id", "source_train_run_id"):
                out[field] = int(value) if value not in (None, "") else None
                continue
            out[field] = "" if value is None else str(value)
        return out

    # -- 옵션 값 사용 현황 ---------------------------------------------------
    _COUNTABLE = {"server", "model", "dataset", "optimizer", "device"}

    def count_runs_using(self, column: str, value: str) -> int:
        """어떤 옵션 값을 쓰는 실행이 몇 건인지 (삭제/이름변경 경고용)."""
        if column not in self._COUNTABLE or not value:
            return 0
        total = 0
        for table, fields in (("train_runs", TRAIN_FIELDS), ("inference_runs", INFER_FIELDS)):
            if column not in fields:
                continue
            row = self.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (value,)
            ).fetchone()
            total += int(row[0])
        return total

    def rename_value_in_runs(self, column: str, old: str, new: str) -> int:
        """옵션 이름을 바꿀 때 기존 기록도 함께 갱신한다."""
        if column not in self._COUNTABLE or not old:
            return 0
        changed = 0
        for table, fields in (("train_runs", TRAIN_FIELDS), ("inference_runs", INFER_FIELDS)):
            if column not in fields:
                continue
            cur = self._exec(
                f"UPDATE {table} SET {column} = ?, updated_at = ? WHERE {column} = ?",
                (new, now_iso(), old),
            )
            changed += cur.rowcount
        return changed

    def count_extra_value(self, field_name: str, value: str) -> int:
        """extra_json 안의 사용자 정의 필드 값 사용 건수."""
        if not field_name or not value:
            return 0
        total = 0
        for table in ("train_runs", "inference_runs"):
            rows = self.conn.execute(f"SELECT extra_json FROM {table}").fetchall()
            for row in rows:
                try:
                    data = json.loads(row[0] or "{}")
                except (TypeError, ValueError):
                    continue
                if isinstance(data, dict) and str(data.get(field_name, "")) == value:
                    total += 1
        return total

    # -- 통계 ---------------------------------------------------------------
    def summary(self) -> dict[str, int]:
        q = self.conn.execute
        return {
            "tasks": int(q("SELECT COUNT(*) FROM tasks").fetchone()[0]),
            "works": int(q("SELECT COUNT(*) FROM works").fetchone()[0]),
            "train": int(q("SELECT COUNT(*) FROM train_runs").fetchone()[0]),
            "inference": int(q("SELECT COUNT(*) FROM inference_runs").fetchone()[0]),
            "running": int(
                q("SELECT COUNT(*) FROM train_runs WHERE status = ?", (C.STATUS_RUNNING,)).fetchone()[0]
            ),
        }

    # -- 백업 ---------------------------------------------------------------
    BACKUP_DIRNAME = "backups"
    BACKUP_SUFFIX = ".bak.db"

    def backup(self, keep: int = 5) -> str | None:
        """DB 파일을 옆 폴더에 타임스탬프로 복사하고, 오래된 백업은 지워 `keep` 개만 남긴다.

        SQLite 파일 하나에 모든 기록이 들어 있으므로 실수 한 번(잘못된 삭제, 깨진 편집)에
        전부 잃을 수 있다. 종료 시점마다 스냅샷을 남겨 두면 최소한의 보험이 된다.
        WAL 모드라 커밋된 내용을 그대로 복사하면 되고, 복사 전에 체크포인트로 -wal 을
        메인 파일에 합쳐 둔다.
        """
        if not os.path.exists(self.path):
            return None
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass

        backup_dir = os.path.join(os.path.dirname(self.path), self.BACKUP_DIRNAME)
        os.makedirs(backup_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(backup_dir, f"{stem}_{timestamp}{self.BACKUP_SUFFIX}")

        try:
            shutil.copy2(self.path, dest)
        except OSError:
            return None

        self._prune_backups(backup_dir, stem, keep)
        return dest

    def _prune_backups(self, backup_dir: str, stem: str, keep: int) -> None:
        pattern = os.path.join(backup_dir, f"{stem}_*{self.BACKUP_SUFFIX}")
        existing = sorted(glob.glob(pattern))  # 타임스탬프가 이름에 있어 사전순 = 시간순
        for old in existing[:-keep] if keep > 0 else existing:
            try:
                os.remove(old)
            except OSError:
                pass

    def list_backups(self) -> list[str]:
        backup_dir = os.path.join(os.path.dirname(self.path), self.BACKUP_DIRNAME)
        stem = os.path.splitext(os.path.basename(self.path))[0]
        pattern = os.path.join(backup_dir, f"{stem}_*{self.BACKUP_SUFFIX}")
        return sorted(glob.glob(pattern), reverse=True)

    def distinct_values(
        self, kind: str, column: str, task_id: int | None = None
    ) -> list[str]:
        """콤보박스 보강용 - 이미 기록에 쓰인 값들.

        `task_id` 를 주면 그 Task 의 실행에서만 모은다. 옵션이 Task 별로 관리되므로,
        범위를 주지 않으면 다른 Task 의 값이 섞여 들어온다.
        """
        table = self._table(kind)
        if column not in self._fields(kind):
            return []
        sql = f"SELECT DISTINCT r.{column} FROM {table} r"
        params: tuple = ()
        if task_id is not None:
            sql += (
                " JOIN works w ON w.id = r.work_id"
                " JOIN tasks t ON t.id = w.task_id"
                " WHERE t.id = ? AND"
            )
            params = (task_id,)
        else:
            sql += " WHERE"
        sql += f" r.{column} IS NOT NULL AND r.{column} != '' ORDER BY r.{column} COLLATE NOCASE"
        return [row[0] for row in self.conn.execute(sql, params).fetchall()]

    # -- 전역 검색 -------------------------------------------------------------
    _SEARCH_COLUMNS: dict[str, tuple[str, ...]] = {
        "train_runs": (
            "server", "model", "dataset", "dataset_path", "result_path",
            "notes", "tags", "failure_reason", "exec_command",
        ),
        "inference_runs": (
            "server", "model", "checkpoint_path", "dataset", "dataset_path",
            "result_path", "notes", "tags", "failure_reason", "exec_command",
        ),
    }

    def search_runs(self, text: str, limit: int = 200) -> list[dict[str, Any]]:
        """Task/Work 이름과 실행의 여러 컬럼을 가로질러 부분일치 검색한다."""
        needle = f"%{text.strip()}%"
        if needle == "%%":
            return []
        out: list[dict[str, Any]] = []
        for table, columns in self._SEARCH_COLUMNS.items():
            kind = "train" if table == "train_runs" else "inference"
            where = " OR ".join(f"r.{c} LIKE ?" for c in columns)
            where += " OR t.name LIKE ? OR w.name LIKE ?"
            sql = (
                f"SELECT r.*, w.name AS work_name, t.name AS task_name, t.id AS task_id, "
                f"'{kind}' AS kind "
                f"FROM {table} r "
                f"JOIN works w ON w.id = r.work_id "
                f"JOIN tasks t ON t.id = w.task_id "
                f"WHERE {where} "
                f"ORDER BY r.id DESC LIMIT ?"
            )
            params = [needle] * len(columns) + [needle, needle, limit]
            out.extend(self._rows(self.conn.execute(sql, params)))
        out.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
        return out[:limit]
