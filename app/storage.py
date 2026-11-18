from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence

MAX_OBJECT_BYTES = 1024 * 1024
REPLICA_SLOTS = (1, 2, 3)
RETAINED_VERSION_REPLICAS = 10
_HASH_CHUNK = 256 * 1024


class StorageError(RuntimeError):
    """存储层可预期的业务错误。"""


class ObjectNotFound(StorageError):
    pass


class VersionMismatch(StorageError):
    pass


class InspectionBusy(StorageError):
    pass


class InvalidUpload(StorageError):
    pass


class RepairSourceInvalid(StorageError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_filename(raw: str | None) -> str:
    """保留可显示的 basename，去掉路径、控制字符和引号。"""
    name = (raw or "").replace("\\", "/")
    name = PurePosixPath(name).name
    name = "".join(ch for ch in name if ch.isprintable() and ch not in '"\r\n')
    name = name.strip()[:200]
    return name or "object.bin"


@dataclass(frozen=True)
class Version:
    id: str
    seq: int
    digest: str
    size: int
    filename: str
    created_at: str
    status: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Version":
        return cls(
            id=row["id"],
            seq=row["seq"],
            digest=row["digest"],
            size=row["size"],
            filename=row["filename"],
            created_at=row["created_at"],
            status=row["status"],
        )


@dataclass(frozen=True)
class ReplicaCheck:
    slot: int
    relative_path: str
    state: str  # verified | corrupt | missing | unreadable
    exists: bool
    observed_size: int | None
    observed_digest: str | None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Inspection:
    id: int
    version_id: str
    version_seq: int
    expected_digest: str
    state: str
    started_at: str
    finished_at: str | None
    summary: str | None
    repaired_count: int
    progress: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Inspection":
        return cls(
            id=row["id"],
            version_id=row["version_id"],
            version_seq=row["version_seq"],
            expected_digest=row["expected_digest"],
            state=row["state"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            summary=row["summary"],
            repaired_count=row["repaired_count"],
            progress=row["progress"],
        )


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


class ObjectStorage:
    """单对象、三副本存储。所有元数据写操作由进程内 RLock 串行化。"""

    def __init__(self, data_dir: str | Path, db_filename: str = "app.db") -> None:
        self.data_dir = Path(data_dir).resolve()
        self.db_path = self.data_dir / db_filename
        self.replica_dirs = {
            slot: self.data_dir / f"replica-{slot}" for slot in REPLICA_SLOTS
        }
        self._write_lock = threading.RLock()
        self._active_inspections: set[int] = set()

    # ------------------------------------------------------------------ setup
    def init(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for directory in self.replica_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            self._create_schema(conn)
            conn.execute(
                "INSERT OR IGNORE INTO objects(id, current_version_id) VALUES (1, NULL)"
            )
            now = utc_now()
            stale = conn.execute(
                "SELECT id, version_seq FROM inspections WHERE state = 'running'"
            ).fetchall()
            for row in stale:
                conn.execute(
                    """
                    UPDATE inspections
                       SET state = 'interrupted', finished_at = ?, progress = NULL,
                           summary = ?
                     WHERE id = ?
                    """,
                    (
                        now,
                        "巡检/修复进程在完成前中断。状态已按磁盘上的实际副本重新核对，"
                        "可再次发起巡检继续修复。",
                        row["id"],
                    ),
                )
                self._log(
                    conn,
                    "warning",
                    "inspection_interrupted",
                    f"巡检 #{row['id']}（版本 #{row['version_seq']}）因进程中断被标记为未完成。",
                    inspection_id=row["id"],
                )

        self._cleanup_temporaries()
        self._remove_orphan_replica_files()
        self.reconcile_current(reason="服务启动")
        self._log_standalone(
            "info", "startup", "服务启动：已核对当前对象和三个副本的实际可读状态。"
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS versions (
                id          TEXT PRIMARY KEY,
                seq         INTEGER NOT NULL UNIQUE,
                digest      TEXT NOT NULL,
                size        INTEGER NOT NULL,
                filename    TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                status      TEXT NOT NULL CHECK(status IN ('current','superseded'))
            );

            CREATE TABLE IF NOT EXISTS objects (
                id                 INTEGER PRIMARY KEY CHECK(id = 1),
                current_version_id TEXT REFERENCES versions(id)
            );

            CREATE TABLE IF NOT EXISTS replicas (
                version_id      TEXT NOT NULL REFERENCES versions(id) ON DELETE CASCADE,
                slot            INTEGER NOT NULL CHECK(slot BETWEEN 1 AND 3),
                relative_path   TEXT NOT NULL,
                last_checked_at TEXT,
                last_check_state TEXT,
                observed_digest TEXT,
                observed_size   INTEGER,
                note            TEXT,
                PRIMARY KEY (version_id, slot)
            );

            CREATE TABLE IF NOT EXISTS inspections (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id      TEXT NOT NULL REFERENCES versions(id),
                version_seq     INTEGER NOT NULL,
                expected_digest TEXT NOT NULL,
                state           TEXT NOT NULL,
                started_at      TEXT NOT NULL,
                finished_at     TEXT,
                summary         TEXT,
                repaired_count  INTEGER NOT NULL DEFAULT 0,
                progress        TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                at             TEXT NOT NULL,
                level          TEXT NOT NULL,
                kind           TEXT NOT NULL,
                message        TEXT NOT NULL,
                version_id     TEXT,
                inspection_id  INTEGER
            );

            CREATE INDEX IF NOT EXISTS ix_inspections_started
                ON inspections(started_at DESC);
            CREATE INDEX IF NOT EXISTS ix_events_at ON events(at DESC);
            """
        )

    def _cleanup_temporaries(self) -> None:
        for path in list(self.replica_dirs.values()):
            for temp in path.glob(".tmp-*"):
                temp.unlink(missing_ok=True)

    def _remove_orphan_replica_files(self) -> None:
        with self._db() as conn:
            rows = conn.execute("SELECT relative_path FROM replicas").fetchall()
        known = {row["relative_path"] for row in rows}
        removed = 0
        for directory in self.replica_dirs.values():
            for path in directory.glob("v-*.bin"):
                rel = path.relative_to(self.data_dir).as_posix()
                if rel not in known:
                    path.unlink(missing_ok=True)
                    removed += 1
        if removed:
            self._log_standalone(
                "warning", "cleanup", f"清理了 {removed} 个未登记版本的中断上传残留文件。"
            )

    # -------------------------------------------------------------- file paths
    def _abs_path(self, relative_path: str) -> Path:
        candidate = (self.data_dir / relative_path).resolve()
        root = self.data_dir.resolve()
        if candidate != root and root not in candidate.parents:
            raise StorageError("非法的副本路径。")
        return candidate

    def _relative_replica_path(self, version_id: str, slot: int) -> str:
        return f"replica-{slot}/v-{version_id}.bin"

    @staticmethod
    def _atomic_write(destination: Path, data: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.parent / f".tmp-{uuid.uuid4().hex}"
        try:
            with temp.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, destination)
            try:
                dir_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        finally:
            temp.unlink(missing_ok=True)

    # ------------------------------------------------------------- read helpers
    def current_version(self) -> Version | None:
        with self._db() as conn:
            row = conn.execute(
                """
                SELECT v.*
                  FROM objects o
                  JOIN versions v ON v.id = o.current_version_id
                 WHERE o.id = 1
                """
            ).fetchone()
        return Version.from_row(row) if row else None

    def get_version(self, version_id: str) -> Version | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM versions WHERE id = ?", (version_id,)
            ).fetchone()
        return Version.from_row(row) if row else None

    def list_versions(self, limit: int = 20) -> list[Version]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM versions ORDER BY seq DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Version.from_row(row) for row in rows]

    def replica_rows(self, version_id: str) -> dict[int, sqlite3.Row]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM replicas WHERE version_id = ?", (version_id,)
            ).fetchall()
        return {row["slot"]: row for row in rows}

    def check_replicas(self, version: Version) -> list[ReplicaCheck]:
        rows = self.replica_rows(version.id)
        checks: list[ReplicaCheck] = []
        for slot in REPLICA_SLOTS:
            row = rows.get(slot)
            if row is None:
                checks.append(
                    ReplicaCheck(slot, "", "missing", False, None, None, "未登记副本路径")
                )
                continue
            rel = row["relative_path"]
            path = self._abs_path(rel)
            try:
                stat = path.stat()
            except FileNotFoundError:
                checks.append(
                    ReplicaCheck(slot, rel, "missing", False, None, None, "副本文件不存在")
                )
                continue
            except OSError as exc:
                checks.append(
                    ReplicaCheck(slot, rel, "unreadable", False, None, None, str(exc))
                )
                continue
            try:
                if stat.st_size > MAX_OBJECT_BYTES:
                    checks.append(
                        ReplicaCheck(
                            slot,
                            rel,
                            "corrupt",
                            True,
                            stat.st_size,
                            None,
                            "副本超过 1 MiB 上限，无法作为有效对象读取",
                        )
                    )
                    continue
                if stat.st_size != version.size:
                    checks.append(
                        ReplicaCheck(
                            slot,
                            rel,
                            "corrupt",
                            True,
                            stat.st_size,
                            None,
                            f"长度不符（应为 {version.size} 字节）",
                        )
                    )
                    continue
                observed_digest, observed_size = _hash_file(path)
            except OSError as exc:
                checks.append(
                    ReplicaCheck(slot, rel, "unreadable", True, None, None, str(exc))
                )
                continue
            if observed_size != version.size:
                checks.append(
                    ReplicaCheck(
                        slot,
                        rel,
                        "corrupt",
                        True,
                        observed_size,
                        observed_digest,
                        "读取过程中长度发生变化",
                    )
                )
            elif observed_digest != version.digest:
                checks.append(
                    ReplicaCheck(
                        slot,
                        rel,
                        "corrupt",
                        True,
                        observed_size,
                        observed_digest,
                        "SHA-256 摘要与当前版本登记值不符",
                    )
                )
            else:
                checks.append(
                    ReplicaCheck(
                        slot,
                        rel,
                        "verified",
                        True,
                        observed_size,
                        observed_digest,
                        None,
                    )
                )
        return checks

    def read_verified_replica(
        self, version: Version, slot: int
    ) -> tuple[bytes, str] | None:
        """读取指定副本，只有完整且摘要匹配登记版本时才返回字节。"""
        rows = self.replica_rows(version.id)
        row = rows.get(slot)
        if row is None:
            return None
        path = self._abs_path(row["relative_path"])
        try:
            if path.stat().st_size != version.size:
                return None
            data = path.read_bytes()
        except OSError:
            return None
        if len(data) != version.size:
            return None
        digest = _hash_bytes(data)
        if digest != version.digest:
            return None
        return data, digest

    def read_current_verified(
        self, expected_version_id: str | None = None
    ) -> tuple[Version | None, bytes | None, int | None]:
        """在写锁内读取版本，并只返回摘要匹配的完整字节。

        未指定 version_id 时读取当前版本；指定时按该登记版本交付（页面“所报版本”）。
        """
        with self._write_lock:
            if expected_version_id:
                version = self.get_version(expected_version_id)
                if version is None:
                    return None, None, None
            else:
                version = self.current_version()
                if version is None:
                    return None, None, None
            for slot in REPLICA_SLOTS:
                result = self.read_verified_replica(version, slot)
                if result is not None:
                    data, _digest = result
                    return version, data, slot
            return version, None, None

    # ------------------------------------------------------------------ writes
    def create_version(self, data: bytes, filename: str) -> Version:
        if len(data) > MAX_OBJECT_BYTES:
            raise InvalidUpload("文件超过 1 MiB 上限。")
        filename = clean_filename(filename)
        digest = _hash_bytes(data)
        version_id = uuid.uuid4().hex
        created_at = utc_now()

        with self._write_lock:
            current = self.current_version()
            with self._db() as conn:
                row = conn.execute("SELECT COALESCE(MAX(seq), 0) AS m FROM versions").fetchone()
                seq = int(row["m"]) + 1
            version = Version(
                id=version_id,
                seq=seq,
                digest=digest,
                size=len(data),
                filename=filename,
                created_at=created_at,
                status="current",
            )
            written: list[Path] = []
            try:
                for slot in REPLICA_SLOTS:
                    rel = self._relative_replica_path(version_id, slot)
                    path = self._abs_path(rel)
                    self._atomic_write(path, data)
                    written.append(path)
                    observed, observed_size = _hash_file(path)
                    if observed != digest or observed_size != len(data):
                        raise StorageError(f"副本 {slot} 写入后校验失败。")
            except BaseException:
                for path in written:
                    path.unlink(missing_ok=True)
                raise

            try:
                with self._db() as conn:
                    # 在同一事务内登记新版本并切换 current 指针。
                    conn.execute(
                        """
                        INSERT INTO versions(id, seq, digest, size, filename, created_at, status)
                        VALUES (?, ?, ?, ?, ?, ?, 'current')
                        """,
                        (version.id, version.seq, version.digest, version.size,
                         version.filename, version.created_at),
                    )
                    for slot in REPLICA_SLOTS:
                        rel = self._relative_replica_path(version_id, slot)
                        conn.execute(
                            """
                            INSERT INTO replicas(
                                version_id, slot, relative_path, last_checked_at,
                                last_check_state, observed_digest, observed_size, note
                            ) VALUES (?, ?, ?, ?, 'verified', ?, ?, NULL)
                            """,
                            (version.id, slot, rel, created_at, digest, len(data)),
                        )
                    conn.execute(
                        "UPDATE versions SET status = 'superseded' WHERE status = 'current'"
                    )
                    conn.execute(
                        "UPDATE versions SET status = 'current' WHERE id = ?", (version.id,)
                    )
                    conn.execute(
                        "UPDATE objects SET current_version_id = ? WHERE id = 1",
                        (version.id,),
                    )
                    self._log(
                        conn,
                        "info",
                        "upload",
                        f"登记版本 #{version.seq}（{len(data)} 字节，SHA-256 {digest}），"
                        "三个副本均已写入并通过摘要校验。",
                        version_id=version.id,
                    )
            except BaseException:
                for path in written:
                    path.unlink(missing_ok=True)
                raise

            old = f"；替换自版本 #{current.seq}" if current else ""
            self._log_standalone(
                "info",
                "replace",
                f"当前对象切换为版本 #{version.seq}{old}。",
                version_id=version.id,
            )
            self._prune_old_versions()
            return version

    def _prune_old_versions(self) -> None:
        """只保留最近若干版本的副本文件，避免磁盘无限增长。

        版本登记与巡检历史保留；被裁掉副本的旧版本下载会返回 409（无有效副本）。
        """
        with self._write_lock:
            with self._db() as conn:
                rows = conn.execute(
                    """
                    SELECT id, seq FROM versions
                    ORDER BY seq DESC LIMIT -1 OFFSET ?
                    """,
                    (RETAINED_VERSION_REPLICAS,),
                ).fetchall()
                removed = 0
                for row in rows:
                    paths = conn.execute(
                        "SELECT relative_path FROM replicas WHERE version_id = ?",
                        (row["id"],),
                    ).fetchall()
                    for path_row in paths:
                        try:
                            path = self._abs_path(path_row["relative_path"])
                        except StorageError:
                            continue
                        if path.exists():
                            path.unlink(missing_ok=True)
                            removed += 1
                    conn.execute(
                        """
                        UPDATE replicas
                           SET last_checked_at = ?, last_check_state = 'missing',
                               observed_digest = NULL, observed_size = NULL,
                               note = '历史版本副本已按保留策略清理'
                         WHERE version_id = ?
                        """,
                        (utc_now(), row["id"]),
                    )
                if removed:
                    self._log(
                        conn,
                        "info",
                        "prune",
                        f"已清理版本 #{rows[-1]['seq']} 及更早版本的 {removed} 个副本文件"
                        f"（保留最近 {RETAINED_VERSION_REPLICAS} 个版本）。",
                    )

    def corrupt_replica(
        self, slot: int, expected_version_id: str | None = None
    ) -> Version:
        if slot not in REPLICA_SLOTS:
            raise StorageError("副本编号必须为 1、2 或 3。")
        with self._write_lock:
            version = self.current_version()
            if version is None:
                raise ObjectNotFound("尚未上传对象。")
            if expected_version_id and expected_version_id != version.id:
                raise VersionMismatch("页面上的版本已过期，对象已被替换。")
            rows = self.replica_rows(version.id)
            row = rows.get(slot)
            if row is None:
                raise StorageError("该副本没有登记路径。")
            path = self._abs_path(row["relative_path"])
            try:
                if path.stat().st_size > MAX_OBJECT_BYTES:
                    tampered = b"corrupted-replica\n"
                else:
                    original = path.read_bytes()
                    tampered = (
                        bytes(byte ^ 0xFF for byte in original)
                        if original
                        else b"corrupted-empty-replica\n"
                    )
            except FileNotFoundError:
                # 副本已缺失时也直接写入损坏内容，模拟“损坏为不可用副本”。
                tampered = b"corrupted-missing-replica\n"
            except OSError as exc:
                raise StorageError(f"读取副本失败：{exc}") from exc
            self._atomic_write(path, tampered)
            with self._db() as conn:
                conn.execute(
                    """
                    UPDATE replicas
                       SET last_checked_at = ?, last_check_state = 'corrupt',
                           observed_digest = ?, observed_size = ?, note = ?
                     WHERE version_id = ? AND slot = ?
                    """,
                    (utc_now(), _hash_bytes(tampered), len(tampered),
                     "值班员手动模拟损坏", version.id, slot),
                )
                self._log(
                    conn,
                    "warning",
                    "simulate_corruption",
                    f"已模拟损坏版本 #{version.seq} 的副本 {slot}。",
                    version_id=version.id,
                )
            return version

    # -------------------------------------------------------------- inspections
    def begin_inspection(self, expected_version_id: str | None = None) -> Inspection:
        with self._write_lock:
            with self._db() as conn:
                row = conn.execute(
                    """
                    SELECT v.*
                      FROM objects o JOIN versions v ON v.id = o.current_version_id
                     WHERE o.id = 1
                    """
                ).fetchone()
                if row is None:
                    raise ObjectNotFound("尚未上传对象，无法巡检。")
                version = Version.from_row(row)
                if expected_version_id and expected_version_id != version.id:
                    raise VersionMismatch("页面上的版本已过期，请刷新后重试。")
                running = conn.execute(
                    "SELECT id FROM inspections WHERE state = 'running' LIMIT 1"
                ).fetchone()
                started = utc_now()
                if self._active_inspections:
                    raise InspectionBusy("已有巡检正在执行，请等待其结束后再试。")
                if running is not None:
                    # 启动时已把上一个进程遗留的 running 标记为 interrupted。
                    # 此处若仍有 running，说明工作线程异常死亡但没来得及落库：
                    # 按中断处理，允许继续修复。
                    conn.execute(
                        """
                        UPDATE inspections
                           SET state = 'interrupted', finished_at = ?, progress = NULL,
                               summary = ?
                         WHERE id = ? AND state = 'running'
                        """,
                        (
                            started,
                            "上一次巡检/修复未正常结束，已按中断处理；"
                            "可重新发起巡检继续修复。",
                            running["id"],
                        ),
                    )
                    self._log(
                        conn,
                        "warning",
                        "inspection_interrupted",
                        f"巡检 #{running['id']} 未正常结束，已标记为中断；开始新的巡检。",
                    )
                cursor = conn.execute(
                    """
                    INSERT INTO inspections(
                        version_id, version_seq, expected_digest, state,
                        started_at, progress
                    ) VALUES (?, ?, ?, 'running', ?, ?)
                    """,
                    (version.id, version.seq, version.digest, started, "巡检已排队。"),
                )
                inspection_id = int(cursor.lastrowid)
                self._log(
                    conn,
                    "info",
                    "inspection_started",
                    f"开始巡检版本 #{version.seq}（SHA-256 {version.digest}）。",
                    version_id=version.id,
                    inspection_id=inspection_id,
                )
                row = conn.execute(
                    "SELECT * FROM inspections WHERE id = ?", (inspection_id,)
                ).fetchone()
        self._active_inspections.add(inspection_id)
        return Inspection.from_row(row)

    def set_inspection_progress(self, inspection_id: int, progress: str) -> None:
        with self._db() as conn:
            conn.execute(
                "UPDATE inspections SET progress = ? WHERE id = ? AND state = 'running'",
                (progress, inspection_id),
            )

    def is_current(self, version_id: str) -> bool:
        version = self.current_version()
        return bool(version and version.id == version_id)

    def release_inspection(self, inspection_id: int) -> None:
        """工作线程退出时释放进程内占位（幂等）。"""
        self._active_inspections.discard(inspection_id)

    def _cache_checks(
        self,
        conn: sqlite3.Connection,
        version_id: str,
        checks: Sequence[ReplicaCheck],
        checked_at: str,
    ) -> None:
        for check in checks:
            conn.execute(
                """
                UPDATE replicas
                   SET last_checked_at = ?, last_check_state = ?,
                       observed_digest = ?, observed_size = ?, note = ?
                 WHERE version_id = ? AND slot = ?
                """,
                (
                    checked_at,
                    check.state,
                    check.observed_digest,
                    check.observed_size,
                    check.note,
                    version_id,
                    check.slot,
                ),
            )

    def finalize_inspection(
        self,
        inspection_id: int,
        version_id: str,
        state: str,
        summary: str,
        checks: Sequence[ReplicaCheck],
        repaired_count: int,
    ) -> str:
        """提交巡检结果。若当前版本已变化，只记录 superseded，不碰新版本。"""
        with self._write_lock:
            with self._db() as conn:
                current = conn.execute(
                    "SELECT current_version_id FROM objects WHERE id = 1"
                ).fetchone()
                current_id = current["current_version_id"] if current else None
                now = utc_now()
                final_state = state
                final_summary = summary
                if current_id != version_id:
                    final_state = "superseded"
                    final_summary = (
                        "巡检期间对象已被替换为新版本。本次巡检结果仅属于旧版本，"
                        "未修改新版本的文件或状态；新版本状态按磁盘实际副本单独计算。"
                    )
                else:
                    self._cache_checks(conn, version_id, checks, now)
                conn.execute(
                    """
                    UPDATE inspections
                       SET state = ?, finished_at = ?, summary = ?,
                           repaired_count = ?, progress = NULL
                     WHERE id = ?
                    """,
                    (final_state, now, final_summary, repaired_count, inspection_id),
                )
                seq_row = conn.execute(
                    "SELECT version_seq FROM inspections WHERE id = ?",
                    (inspection_id,),
                ).fetchone()
                version_seq = seq_row["version_seq"] if seq_row else "?"
                if final_state == "superseded":
                    message = f"巡检 #{inspection_id}（版本 #{version_seq}）因对象被替换而作废。"
                    level = "warning"
                elif final_state == "completed":
                    message = f"巡检 #{inspection_id}（版本 #{version_seq}）完成：{final_summary}"
                    level = "info"
                elif final_state == "unrecoverable":
                    message = f"巡检 #{inspection_id}（版本 #{version_seq}）不可恢复：{final_summary}"
                    level = "error"
                else:
                    message = f"巡检 #{inspection_id}（版本 #{version_seq}）结束：{final_summary}"
                    level = "warning"
                self._log(
                    conn,
                    level,
                    "inspection_finished",
                    message,
                    version_id=version_id,
                    inspection_id=inspection_id,
                )
        self._active_inspections.discard(inspection_id)
        return final_state

    def mark_inspection_failed(self, inspection_id: int, error: str) -> None:
        with self._write_lock:
            with self._db() as conn:
                conn.execute(
                    """
                    UPDATE inspections
                       SET state = 'failed', finished_at = ?, progress = NULL, summary = ?
                     WHERE id = ? AND state = 'running'
                    """,
                    (utc_now(), f"巡检执行异常：{error}", inspection_id),
                )
                self._log(
                    conn,
                    "error",
                    "inspection_failed",
                    f"巡检 #{inspection_id} 执行异常：{error}",
                    inspection_id=inspection_id,
                )
        self._active_inspections.discard(inspection_id)

    def stage_repair(
        self,
        version: Version,
        source_slot: int,
        target_slot: int,
    ) -> tuple[Path, str]:
        """从已验真副本读取字节，写入目标副本目录内的临时文件并再次校验。"""
        result = self.read_verified_replica(version, source_slot)
        if result is None:
            raise RepairSourceInvalid(f"副本 {source_slot} 未能通过当前版本摘要校验，不能作为修复源。")
        data, digest = result
        rows = self.replica_rows(version.id)
        target_row = rows.get(target_slot)
        if target_row is None:
            raise StorageError(f"副本 {target_slot} 未登记。")
        target = self._abs_path(target_row["relative_path"])
        temp = target.parent / f".tmp-repair-{version.id}-{target_slot}-{uuid.uuid4().hex}"
        try:
            with temp.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            observed, observed_size = _hash_file(temp)
            if observed != version.digest or observed_size != version.size:
                raise RepairSourceInvalid("修复临时文件摘要校验失败。")
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        return temp, digest

    def commit_repair(
        self,
        version_id: str,
        target_slot: int,
        staged_path: Path,
        source_slot: int,
        expected_digest: str,
    ) -> str:
        """锁内再次检查版本仍未变化，然后原子替换目标副本。"""
        with self._write_lock:
            with self._db() as conn:
                row = conn.execute(
                    "SELECT current_version_id FROM objects WHERE id = 1"
                ).fetchone()
                current_id = row["current_version_id"] if row else None
            if current_id != version_id:
                staged_path.unlink(missing_ok=True)
                return "stale"

            try:
                digest, size = _hash_file(staged_path)
                if digest != expected_digest:
                    raise RepairSourceInvalid(
                        "修复暂存文件在提交前摘要不符，已放弃本次修复。"
                    )
                version_row = self.get_version(version_id)
                if version_row is None:
                    return "stale"
                if size != version_row.size:
                    raise RepairSourceInvalid("修复暂存文件长度不符，已放弃本次修复。")
                rows = self.replica_rows(version_id)
                target_row = rows.get(target_slot)
                if target_row is None:
                    raise StorageError("目标副本没有登记路径。")
                target = self._abs_path(target_row["relative_path"])
                if staged_path.parent != target.parent:
                    raise StorageError("修复暂存文件与目标副本不在同一目录。")
                # 提交前最后一次确认当前版本指针未变化，避免把旧字节盖到新版本上。
                if not self.is_current(version_id):
                    return "stale"
                os.replace(staged_path, target)
                try:
                    dir_fd = os.open(target.parent, os.O_RDONLY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except OSError:
                    pass
                observed, observed_size = _hash_file(target)
                if observed != expected_digest or observed_size != size:
                    raise RepairSourceInvalid("修复后的副本校验失败。")
            finally:
                staged_path.unlink(missing_ok=True)

            now = utc_now()
            with self._db() as conn:
                conn.execute(
                    """
                    UPDATE replicas
                       SET last_checked_at = ?, last_check_state = 'verified',
                           observed_digest = ?, observed_size = ?, note = ?
                     WHERE version_id = ? AND slot = ?
                    """,
                    (
                        now,
                        expected_digest,
                        size,
                        f"由已验真副本 {source_slot} 复制修复",
                        version_id,
                        target_slot,
                    ),
                )
                version_row = conn.execute(
                    "SELECT seq FROM versions WHERE id = ?", (version_id,)
                ).fetchone()
                seq = version_row["seq"] if version_row else "?"
                self._log(
                    conn,
                    "info",
                    "replica_repaired",
                    f"版本 #{seq} 的副本 {target_slot} 已从已验真副本 {source_slot} 复制修复。",
                    version_id=version_id,
                )
            return "committed"

    def reconcile_current(self, reason: str) -> list[ReplicaCheck]:
        with self._write_lock:
            version = self.current_version()
            if version is None:
                return []
            checks = self.check_replicas(version)
            with self._db() as conn:
                self._cache_checks(conn, version.id, checks, utc_now())
                verified = sum(1 for check in checks if check.state == "verified")
                self._log(
                    conn,
                    "info" if verified == 3 else "warning",
                    "reconcile",
                    f"{reason}：版本 #{version.seq} 实际可读副本 {verified}/3。",
                    version_id=version.id,
                )
            return checks

    # --------------------------------------------------------------- reporting
    def recent_inspections(self, limit: int = 10) -> list[Inspection]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Inspection.from_row(row) for row in rows]

    def running_inspection(self) -> Inspection | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT * FROM inspections WHERE state = 'running' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return Inspection.from_row(row) if row else None

    def recent_events(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------- audit
    @staticmethod
    def _log(
        conn: sqlite3.Connection,
        level: str,
        kind: str,
        message: str,
        *,
        version_id: str | None = None,
        inspection_id: int | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO events(at, level, kind, message, version_id, inspection_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (utc_now(), level, kind, message, version_id, inspection_id),
        )

    def _log_standalone(
        self,
        level: str,
        kind: str,
        message: str,
        *,
        version_id: str | None = None,
        inspection_id: int | None = None,
    ) -> None:
        with self._db() as conn:
            self._log(
                conn,
                level,
                kind,
                message,
                version_id=version_id,
                inspection_id=inspection_id,
            )
