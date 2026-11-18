from __future__ import annotations

import io
import threading
from pathlib import Path
from urllib.parse import unquote

import pytest
from starlette.testclient import TestClient

from app.main import create_app
from app.maintenance import run_inspection
from app.storage import (
    ObjectStorage,
    QuotaExceeded,
    StorageError,
)


@pytest.fixture()
def app_env(tmp_path: Path):
    app = create_app(tmp_path / "data", start_background=False)
    client = TestClient(app)
    with client:
        yield client, app.state.storage


def upload(client: TestClient, data: bytes, filename: str = "payload.bin"):
    return client.post(
        "/upload",
        files={"file": (filename, io.BytesIO(data), "application/octet-stream")},
        follow_redirects=False,
    )


def replica_path(storage: ObjectStorage, slot: int, version_id: str | None = None) -> Path:
    if version_id is None:
        version = storage.current_version()
        assert version is not None
        version_id = version.id
    row = storage.replica_rows(version_id)[slot]
    return storage.data_dir / row["relative_path"]


def count_temp_files(storage: ObjectStorage) -> int:
    return sum(1 for _ in storage.data_dir.glob("**/.tmp-*"))


def test_quota_blocks_upload_keeps_old_version_and_flags_directory(app_env):
    client, storage = app_env
    payload = b"a" * 100
    assert upload(client, payload).status_code == 303
    version = storage.current_version()
    assert version is not None and version.size == 100

    # 把三个目录上限都压到当前占用：页面应报告目录被阻塞。
    for slot in (1, 2, 3):
        storage.set_quota(slot, 100)
    page = client.get("/").text
    assert "副本目录容量" in page
    assert "已阻塞" in page
    report = {item.slot: item for item in storage.quota_report()}
    assert all(item.full for item in report.values())

    # 直接调存储层：阻塞异常必须指出具体目录与账本数字。
    with pytest.raises(QuotaExceeded) as exc_info:
        storage.create_version(b"b" * 150, "bigger.bin")
    assert exc_info.value.slot == 1
    assert exc_info.value.quota == 100
    assert exc_info.value.used == 100
    assert exc_info.value.needed == 150

    # 网页上传同样被拒：旧版本保持可读，没有暂存残留，版本指针未变化。
    response = upload(client, b"b" * 150)
    assert response.status_code == 303
    location = unquote(response.headers["location"])
    assert "空间不足" in location and "replica-1" in location
    assert storage.current_version().id == version.id
    assert client.get("/download").content == payload
    assert count_temp_files(storage) == 0

    # 调高上限后允许重试成功。
    for slot in (1, 2, 3):
        storage.set_quota(slot, 1024)
    assert upload(client, b"b" * 150).status_code == 303
    assert client.get("/download").content == b"b" * 150


def test_usage_counts_staging_and_residual_files(app_env):
    client, storage = app_env
    payload = b"x" * 500
    upload(client, payload)

    # 未清理的暂存文件与任意残留文件都必须计入目录实际占用。
    tmp_residual = storage.replica_dirs[2] / ".tmp-repair-leftover"
    tmp_residual.write_bytes(b"t" * 120)
    other_residual = storage.replica_dirs[3] / "residual.bin"
    other_residual.write_bytes(b"r" * 80)

    report = {item.slot: item for item in storage.quota_report()}
    assert report[2].used_bytes == 500 + 120
    assert report[3].used_bytes == 500 + 80
    assert report[1].used_bytes == 500

    # 清理端点释放暂存空间；非暂存残留仍计入，直到被启动清理或人工删除。
    response = client.post("/maintenance/cleanup", data={}, follow_redirects=False)
    assert response.status_code == 303
    assert not tmp_residual.exists()
    assert other_residual.exists()
    report = {item.slot: item for item in storage.quota_report()}
    assert report[2].used_bytes == 500
    assert report[3].used_bytes == 580


def test_interleaved_operations_cannot_double_book_free_space(app_env):
    client, storage = app_env
    size = 300
    upload(client, b"p" * size)
    version = storage.current_version()

    # 每个目录只放得下“现有副本 + 一份在途写入”。
    for slot in (1, 2, 3):
        storage.set_quota(slot, 2 * size)

    # 模拟一个正在进行的上传先预留了副本 2 的空闲空间。
    with storage._reserve_space(2, size):
        assert storage._reserved_bytes[2] == size
        # 巡检修复想往同一目录暂存：同一块空闲空间不能再用，必须被配额挡住。
        replica_path(storage, 2).write_bytes(b"bad")
        with pytest.raises(QuotaExceeded) as exc_info:
            storage.stage_repair(version, 1, 2)
        assert exc_info.value.slot == 2

        # 新上传在三个目录整体预留，任一目录（此处为 2）不够就整体拒绝。
        with pytest.raises(QuotaExceeded):
            storage.create_version(b"q" * size, "q.bin")
        assert count_temp_files(storage) == 0

    # 预留释放后，修复暂存与上传都可重试成功。
    staged, digest = storage.stage_repair(version, 1, 2)
    try:
        report = {item.slot: item for item in storage.quota_report()}
        # 暂存文件落在同目录：副本 2 原文件已被 3 字节坏数据替换，加上 300 字节暂存。
        assert report[2].used_bytes == 3 + size
    finally:
        staged.unlink()
    assert upload(client, b"q" * size).status_code == 303


def test_inspection_fails_not_success_when_quota_full_then_retry(app_env):
    client, storage = app_env
    size = 400
    payload = b"z" * size
    upload(client, payload)
    version = storage.current_version()
    replica_path(storage, 2).write_bytes(b"b" * size)
    replica_path(storage, 3).write_bytes(b"b" * size)

    # 副本 2、3 目录已无空闲：修复暂存必然超出上限。
    storage.set_quota(2, size)
    storage.set_quota(3, size)

    client.post("/inspect", data={}, follow_redirects=False)
    latest = storage.recent_inspections(1)[0]
    assert latest.state == "failed"
    assert "空间不足" in (latest.summary or "")
    # 绝不报修复成功；原本可读的副本仍可下载。
    assert client.get("/download").content == payload
    status = client.get("/api/status").json()
    assert status["readable"] == 1
    assert count_temp_files(storage) == 0

    # 释放空间（调高上限）后再次巡检，修复才真正完成。
    storage.set_quota(2, 4 * size)
    storage.set_quota(3, 4 * size)
    client.post("/inspect", data={}, follow_redirects=False)
    status = client.get("/api/status").json()
    assert status["readable"] == 3
    assert status["latest_inspection"]["state"] == "completed"
    assert client.get("/download").content == payload


def test_threaded_inspection_and_upload_respect_one_reservation_ledger(
    app_env, monkeypatch
):
    client, storage = app_env
    size = 250
    upload(client, b"o" * size)
    version = storage.current_version()
    replica_path(storage, 3).write_bytes(b"bad")
    for slot in (1, 2, 3):
        storage.set_quota(slot, 2 * size)

    real_reserve = ObjectStorage._reserve_space
    barrier = threading.Event()
    release = threading.Event()

    def paused_reserve(self, slot, needed):
        context = real_reserve(self, slot, needed)
        # 进入上下文体（预留已记账）之后再暂停，模拟修复暂存进行到一半。
        class _Paused:
            def __enter__(self_inner):
                self_inner._cm = context.__enter__()
                if slot == 3:
                    barrier.set()
                    assert release.wait(5), "测试同步超时"
                return self_inner._cm

            def __exit__(self_inner, *exc):
                return context.__exit__(*exc)

        return _Paused()

    monkeypatch.setattr(ObjectStorage, "_reserve_space", paused_reserve)

    inspection = storage.begin_inspection(version.id)
    worker = threading.Thread(
        target=run_inspection, args=(storage, inspection.id, version.id)
    )
    worker.start()
    assert barrier.wait(5)
    try:
        # 巡检已在副本 3 持有暂存预留（250 字节）；同尺寸上传在该目录没有
        # 第二份空闲空间，必须被同一本预留账本挡住，不能重复使用同一块空间。
        assert storage._reserved_bytes[3] == size
        with pytest.raises(QuotaExceeded) as exc_info:
            storage.create_version(b"n" * size, "n.bin")
        assert exc_info.value.slot == 3
        assert count_temp_files(storage) == 0
    finally:
        release.set()
        worker.join(10)
    monkeypatch.undo()
    assert not worker.is_alive()
    assert storage.recent_inspections(1)[0].state == "completed"

    # 巡检结束、预留归还后，上传可成功。
    assert upload(client, b"n" * size).status_code == 303


def test_restart_recomputes_usage_and_persists_quotas(tmp_path: Path):
    data_dir = tmp_path / "data"
    storage = ObjectStorage(data_dir)
    storage.init()
    version = storage.create_version(b"r" * 300, "r.bin")

    storage.set_quota(1, 12345)
    storage.set_quota(2, 67890)
    # 模拟崩溃残留：一个暂存文件、一个未登记普通残留。
    (data_dir / "replica-2" / ".tmp-repair-crashed").write_bytes(b"t" * 55)
    (data_dir / "replica-3" / "residual.bin").write_bytes(b"u" * 77)

    restarted = ObjectStorage(data_dir)
    restarted.init()

    # 配额持久化，不依赖进程内缓存。
    quotas = restarted.get_quotas()
    assert quotas[1] == 12345
    assert quotas[2] == 67890

    # 启动清理删掉 .tmp-*；占用按实际文件重新扫描，残留仍计入。
    assert not (data_dir / "replica-2" / ".tmp-repair-crashed").exists()
    report = {item.slot: item for item in restarted.quota_report()}
    assert report[1].used_bytes == 300
    assert report[2].used_bytes == 300
    assert report[3].used_bytes == 300 + 77

    # 重启后页面与接口仍如实显示容量账。
    app = create_app(data_dir, start_background=False)
    client = TestClient(app)
    with client:
        status = client.get("/api/status").json()
        assert len(status["quotas"]) == 3
        by_slot = {item["slot"]: item for item in status["quotas"]}
        assert by_slot[1]["quota_bytes"] == 12345
        assert by_slot[3]["used_bytes"] == 377
        assert client.get("/download").content == b"r" * 300
    assert version is not None


def test_disk_write_failure_preserves_old_version_and_cleans_temp(app_env, monkeypatch):
    client, storage = app_env
    payload = b"keep-me" * 50
    upload(client, payload)
    old = storage.current_version()

    def failing_write(self, destination, data):
        raise OSError("ENOSPC simulated")

    monkeypatch.setattr(ObjectStorage, "_atomic_write", failing_write)
    with pytest.raises(StorageError) as exc_info:
        storage.create_version(b"new" * 50, "new.bin")
    assert "写盘失败" in str(exc_info.value)
    monkeypatch.undo()

    assert storage.current_version().id == old.id
    assert client.get("/download").content == payload
    assert count_temp_files(storage) == 0
    # 失败后系统仍可正常写入。
    assert upload(client, b"recovered" * 10).status_code == 303


def test_quota_endpoints_validation_and_status_payload(app_env):
    client, storage = app_env
    upload(client, b"m" * 10)

    status = client.get("/api/status").json()
    assert [item["slot"] for item in status["quotas"]] == [1, 2, 3]
    first = status["quotas"][0]
    for key in ("used_bytes", "reserved_bytes", "quota_bytes", "available_bytes", "full"):
        assert key in first

    page = client.get("/").text
    assert "副本目录容量" in page and "replica-1" in page

    ok = client.post("/quotas", data={"slot": "2", "quota_bytes": "999"}, follow_redirects=False)
    assert ok.status_code == 303
    assert storage.get_quotas()[2] == 999

    bad_slot = client.post("/quotas", data={"slot": "9", "quota_bytes": "1000"}, follow_redirects=False)
    assert "1、2 或 3" in unquote(bad_slot.headers["location"])
    bad_value = client.post("/quotas", data={"slot": "1", "quota_bytes": "0"}, follow_redirects=False)
    assert "正整数" in unquote(bad_value.headers["location"])
    assert storage.get_quotas()[1] != 0

    cleanup = client.post("/maintenance/cleanup", data={}, follow_redirects=False)
    assert cleanup.status_code == 303


def test_env_var_quota_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REPLICA_QUOTA_BYTES", "5000")
    monkeypatch.setenv("REPLICA_2_QUOTA_BYTES", "9000")
    storage = ObjectStorage(tmp_path / "data")
    storage.init()
    quotas = storage.get_quotas()
    assert quotas[1] == 5000
    assert quotas[2] == 9000
    assert quotas[3] == 5000

    # 落库后即使环境变量变化，重启仍以数据库里的值班设置为准。
    monkeypatch.setenv("REPLICA_QUOTA_BYTES", "1")
    again = ObjectStorage(tmp_path / "data")
    again.init()
    assert again.get_quotas()[1] == 5000
    assert again.get_quotas()[2] == 9000
