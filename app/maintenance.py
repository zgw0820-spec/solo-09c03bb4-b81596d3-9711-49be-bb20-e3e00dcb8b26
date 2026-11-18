from __future__ import annotations

import logging
from typing import Sequence

from .storage import ObjectStorage, RepairSourceInvalid, ReplicaCheck

logger = logging.getLogger(__name__)


def _describe_bad(checks: Sequence[ReplicaCheck]) -> str:
    parts = []
    for check in checks:
        if check.state == "verified":
            continue
        reason = {
            "corrupt": "损坏",
            "missing": "缺失",
            "unreadable": "不可读",
        }.get(check.state, check.state)
        parts.append(f"副本 {check.slot} {reason}")
    return "、".join(parts) if parts else "无异常"


def run_inspection(
    storage: ObjectStorage, inspection_id: int, version_id: str
) -> None:
    """执行一次巡检/修复。

    整个任务钉在开始时的 version_id 上：期间若对象被替换，最终结果只会
    记为 superseded，绝不会更新新版本的副本状态或把新版本标成“已修复”。
    """
    version = storage.get_version(version_id)
    if version is None:
        storage.mark_inspection_failed(inspection_id, "巡检对应的版本记录不存在。")
        storage.release_inspection(inspection_id)
        return

    try:
        storage.set_inspection_progress(inspection_id, "正在逐个读取并校验三个副本……")
        checks = storage.check_replicas(version)
        initial_verified = [check.slot for check in checks if check.state == "verified"]
        storage.set_inspection_progress(
            inspection_id,
            f"初检完成：可读副本 {len(initial_verified)}/3。"
            + ("准备从已验真副本复制修复。" if initial_verified else ""),
        )

        if not initial_verified:
            storage.finalize_inspection(
                inspection_id,
                version.id,
                "unrecoverable",
                "不可恢复：三个副本均未通过当前版本摘要校验（"
                + _describe_bad(checks)
                + "），没有可用于修复的已验真副本。",
                checks,
                0,
            )
            return

        repaired = 0
        verified_slots = set(initial_verified)
        for check in checks:
            if check.state == "verified":
                continue
            if not storage.is_current(version.id):
                break
            storage.set_inspection_progress(
                inspection_id,
                f"正在修复副本 {check.slot}：从已验真副本复制……",
            )
            committed = False
            for source_slot in sorted(verified_slots):
                try:
                    staged_path, digest = storage.stage_repair(
                        version, source_slot, check.slot
                    )
                except RepairSourceInvalid as exc:
                    logger.warning("repair source %s invalid: %s", source_slot, exc)
                    continue
                outcome = storage.commit_repair(
                    version.id, check.slot, staged_path, source_slot, digest
                )
                if outcome == "committed":
                    repaired += 1
                    verified_slots.add(check.slot)
                    committed = True
                    break
                if outcome == "stale":
                    break
            if not committed:
                break

        final_checks = storage.check_replicas(version)
        final_verified = sum(1 for check in final_checks if check.state == "verified")

        if final_verified == 3:
            if repaired:
                state = "completed"
                summary = (
                    f"巡检完成：从已验真副本复制修复了 {repaired} 个副本，"
                    "三个副本现均通过摘要校验。"
                )
            else:
                state = "completed"
                summary = "巡检完成：三个副本均通过摘要校验，无需修复。"
        elif final_verified == 0:
            state = "unrecoverable"
            summary = (
                "不可恢复：三个副本均未通过当前版本摘要校验（"
                + _describe_bad(final_checks)
                + "）。"
            )
        else:
            state = "partial"
            summary = (
                f"巡检结束：仍有 {3 - final_verified} 个副本未通过校验（"
                + _describe_bad(final_checks)
                + f"），可读副本 {final_verified}/3。可再次巡检继续修复。"
            )

        storage.finalize_inspection(
            inspection_id,
            version.id,
            state,
            summary,
            final_checks,
            repaired,
        )
    except Exception as exc:  # noqa: BLE001 - 后台任务必须落库失败状态
        logger.exception("inspection %s failed", inspection_id)
        storage.mark_inspection_failed(inspection_id, str(exc))
    finally:
        storage.release_inspection(inspection_id)
