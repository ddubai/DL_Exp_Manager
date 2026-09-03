"""SQLite 데이터 계층.

로컬 디렉토리에 `experiments.db` 를 만들고 Task -> Work -> Run 계층을 관리한다.

    tasks (Level 1: DL Task)
      └─ works (Level 2: Work ID)
           ├─ train_runs
           └─ inference_runs
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Iterable, Sequence

from . import constants as C
from .utils import dumps_metrics, now_iso

SCHEMA_VERSION = 2

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

CREATE INDEX IF NOT EXISTS idx_works_task       ON works(task_id);
CREATE INDEX IF NOT EXISTS idx_train_work       ON train_runs(work_id);
CREATE INDEX IF NOT EXISTS idx_train_status     ON train_runs(status);
CREATE INDEX IF NOT EXISTS idx_infer_work       ON inference_runs(work_id);
"""

TRAIN_FIELDS: tuple[str, ...] = (
    "work_id", "server", "model", "dataset", "dataset_path", "result_path",
    "status", "started_at", "duration_sec", "epochs", "batch_size", "lr",
    "optimizer", "gpu_indices", "extra_json", "metrics_json", "exec_command",
    "config_yaml", "notes",
)

INFER_FIELDS: tuple[str, ...] = (
    "work_id", "server", "model", "checkpoint_path", "dataset", "dataset_path",
    "result_path", "device", "input_size", "latency_ms", "throughput_fps",
    "status", "started_at", "duration_sec", "gpu_indices", "extra_json",
    "metrics_json", "exec_command", "config_yaml", "notes",
)

# v1 -> v2 에서 추가된 컬럼. (테이블, 컬럼, 정의)
_V2_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("train_runs", "gpu_indices", "TEXT DEFAULT ''"),
    ("train_runs", "extra_json", "TEXT DEFAULT '{}'"),
    ("inference_runs", "gpu_indices", "TEXT DEFAULT ''"),
    ("inference_runs", "extra_json", "TEXT DEFAULT '{}'"),
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
            for table, column, definition in _V2_COLUMNS:
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

    def insert_run(self, kind: str, data: dict[str, Any]) -> int:
        table, fields = self._table(kind), self._fields(kind)
        payload = self._normalize(data, fields)
        ts = now_iso()
        cols = list(fields) + ["created_at", "updated_at"]
        values = [payload.get(f) for f in fields] + [ts, ts]
        placeholders = ",".join("?" * len(cols))
        cur = self._exec(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})", values
        )
        return int(cur.lastrowid)

    def update_run(self, kind: str, run_id: int, data: dict[str, Any]) -> None:
        table, fields = self._table(kind), self._fields(kind)
        payload = self._normalize(data, fields)
        assignments = ",".join(f"{f} = ?" for f in fields)
        values = [payload.get(f) for f in fields] + [now_iso(), run_id]
        self._exec(f"UPDATE {table} SET {assignments}, updated_at = ? WHERE id = ?", values)

    def delete_runs(self, kind: str, run_ids: Iterable[int]) -> int:
        ids = [int(i) for i in run_ids]
        if not ids:
            return 0
        table = self._table(kind)
        placeholders = ",".join("?" * len(ids))
        cur = self._exec(f"DELETE FROM {table} WHERE id IN ({placeholders})", ids)
        return cur.rowcount

    def duplicate_run(self, kind: str, run_id: int) -> int | None:
        row = self.get_run(kind, run_id)
        if row is None:
            return None
        row = dict(row)
        row["notes"] = (row.get("notes") or "").strip()
        row["notes"] = (row["notes"] + "\n(copy)").strip()
        row["status"] = C.STATUS_QUEUED
        return self.insert_run(kind, row)

    @staticmethod
    def _normalize(data: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
        """폼/딕셔너리 값을 DB 컬럼 타입에 맞게 정리."""
        out: dict[str, Any] = {}
        numeric = {"duration_sec", "latency_ms", "throughput_fps"}
        for field in fields:
            value = data.get(field)
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
            if field == "work_id":
                out[field] = int(value) if value is not None else None
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
