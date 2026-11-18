from __future__ import annotations

import mimetypes
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .maintenance import run_inspection
from .storage import (
    InspectionBusy,
    InvalidUpload,
    MAX_OBJECT_BYTES,
    ObjectNotFound,
    ObjectStorage,
    StorageError,
    VersionMismatch,
    utc_now,
)

BASE_DIR = Path(__file__).resolve().parent
MAX_REQUEST_BODY = MAX_OBJECT_BYTES + 512 * 1024
REPLICA_STATE_LABELS = {
    "verified": "已验真",
    "corrupt": "损坏",
    "missing": "缺失",
    "unreadable": "不可读",
}


class _BodyTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """纯 ASGI 中间件：在 receive 层限制请求体，超限直接返回 413。

    仅限制写请求；Content-Length 已知时立即拒绝，chunked 上传按实际读取字节累计。
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BODY) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method", "").upper() not in {
            "POST",
            "PUT",
            "PATCH",
        }:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        declared = headers.get("content-length")
        if declared is not None:
            try:
                too_large = int(declared) > self.max_bytes
            except ValueError:
                await _plain_413(send, "Content-Length 无效。", status=400)
                return
            if too_large:
                await _plain_413(send, "请求体超过 1 MiB 上传上限。")
                return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _BodyTooLarge:
            if not response_started:
                await _plain_413(send, "请求体超过 1 MiB 上传上限。")


async def _plain_413(send: Send, message: str, *, status: int = 413) -> None:
    body = message.encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _safe_download_name(filename: str) -> str:
    fallback = "".join(
        ch if 32 <= ord(ch) < 127 and ch not in '"\\' else "_" for ch in filename
    ).strip() or "object.bin"
    return (
        f'attachment; filename="{fallback}"; '
        f"filename*=UTF-8''{quote(filename, safe='')}"
    )


def create_app(
    data_dir: str | os.PathLike[str] | None = None,
    *,
    runner: Callable[[ObjectStorage, int, str], None] = run_inspection,
    start_background: bool = True,
) -> FastAPI:
    storage = ObjectStorage(data_dir or os.environ.get("DATA_DIR", "/data"))
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await run_in_threadpool(storage.init)
        yield

    app = FastAPI(
        title="值班台 · 三副本单对象",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.storage = storage
    app.state.runner = runner
    app.state.start_background = start_background

    app.add_middleware(BodyLimitMiddleware)
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, notice: str | None = None, level: str = "info"):
        version = storage.current_version()
        checks = storage.check_replicas(version) if version else []
        verified = sum(1 for check in checks if check.state == "verified")
        running = storage.running_inspection()
        ctx: dict[str, Any] = {
            "request": request,
            "version": version,
            "checks": checks,
            "state_labels": REPLICA_STATE_LABELS,
            "verified_count": verified,
            "unrecoverable": bool(version and verified == 0),
            "running": running,
            "inspections": storage.recent_inspections(10),
            "versions": storage.list_versions(10),
            "events": storage.recent_events(25),
            "notice": notice,
            "notice_level": level,
            "max_bytes": MAX_OBJECT_BYTES,
            "now": utc_now(),
        }
        return templates.TemplateResponse(request, "index.html", ctx)

    @app.post("/upload")
    async def upload(request: Request):
        form = None
        try:
            form = await request.form()
            upload_file = form.get("file")
            if upload_file is None or not hasattr(upload_file, "read"):
                return _redirect_notice("请选择要上传的文件。", "error")

            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = await upload_file.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_OBJECT_BYTES:
                    return _redirect_notice(
                        "文件超过 1 MiB 上限，未做任何替换。", "error"
                    )
                chunks.append(chunk)
            data = b"".join(chunks)
            filename = getattr(upload_file, "filename", "") or ""
            version = await run_in_threadpool(storage.create_version, data, filename)
        except InvalidUpload as exc:
            return _redirect_notice(str(exc), "error")
        except StorageError as exc:
            return _redirect_notice(str(exc), "error")
        except _BodyTooLarge:
            raise
        except Exception:
            return _redirect_notice("上传或登记失败，未改变当前对象。", "error")
        finally:
            if form is not None:
                await form.close()
        message = (
            f"已登记版本 #{version.seq}，SHA-256 {version.digest}，"
            "三个副本均写入并验真。"
        )
        return _redirect_notice(message, "success")

    @app.post("/replicas/{slot}/corrupt")
    async def corrupt(slot: int, request: Request):
        form = await request.form()
        try:
            expected = str(form.get("version_id") or "") or None
        finally:
            await form.close()
        try:
            version = await run_in_threadpool(storage.corrupt_replica, slot, expected)
        except (VersionMismatch, ObjectNotFound, StorageError) as exc:
            return _redirect_notice(str(exc), "error")
        return _redirect_notice(
            f"已模拟损坏版本 #{version.seq} 的副本 {slot}，可发起巡检修复。",
            "warning",
        )

    @app.post("/inspect")
    async def inspect(request: Request):
        form = await request.form()
        try:
            expected = str(form.get("version_id") or "") or None
        finally:
            await form.close()
        try:
            inspection = await run_in_threadpool(storage.begin_inspection, expected)
        except (VersionMismatch, InspectionBusy, ObjectNotFound, StorageError) as exc:
            return _redirect_notice(str(exc), "error")

        if app.state.start_background:
            threading.Thread(
                target=app.state.runner,
                args=(storage, inspection.id, inspection.version_id),
                name=f"inspection-{inspection.id}",
                daemon=True,
            ).start()
        else:
            await run_in_threadpool(
                app.state.runner,
                storage,
                inspection.id,
                inspection.version_id,
            )
        return _redirect_notice(f"已发起巡检 #{inspection.id}。", "info")

    @app.get("/download")
    def download(version_id: str | None = None):
        version, data, slot = storage.read_current_verified(version_id)
        if version is None:
            return PlainTextResponse("没有所报版本的登记记录。", status_code=404)
        if data is None:
            return PlainTextResponse(
                "无法交付：当前版本没有任何副本与所报 SHA-256 摘要相符，"
                "已拒绝发送不完整字节。请巡检或重新上传。",
                status_code=409,
            )
        media_type = (
            mimetypes.guess_type(version.filename)[0] or "application/octet-stream"
        )
        headers = {
            "Content-Disposition": _safe_download_name(version.filename),
            "ETag": f'"sha256:{version.digest}"',
            "X-Object-Version": str(version.seq),
            "X-Object-Version-Id": version.id,
            "X-Object-SHA256": version.digest,
            "X-Served-Replica": str(slot),
            "Cache-Control": "no-store",
            "Content-Length": str(len(data)),
        }
        return Response(content=data, media_type=media_type, headers=headers)

    @app.get("/api/status")
    def api_status():
        version = storage.current_version()
        running = storage.running_inspection()
        checks = storage.check_replicas(version) if version else []
        verified = sum(1 for check in checks if check.state == "verified")
        latest = storage.recent_inspections(1)
        return JSONResponse(
            {
                "version_id": version.id if version else None,
                "version_seq": version.seq if version else None,
                "digest": version.digest if version else None,
                "readable": verified,
                "unrecoverable": bool(version and verified == 0),
                "replicas": [check.as_dict() for check in checks],
                "running": (
                    {"id": running.id, "progress": running.progress}
                    if running
                    else None
                ),
                "latest_inspection": (
                    {
                        "id": latest[0].id,
                        "state": latest[0].state,
                        "summary": latest[0].summary,
                    }
                    if latest
                    else None
                ),
            }
        )

    return app


def _redirect_notice(message: str, level: str = "info") -> RedirectResponse:
    return RedirectResponse(
        f"/?notice={quote(message)}&level={quote(level)}", status_code=303
    )


app = create_app()
