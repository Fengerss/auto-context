"""Tag 持久化存储。

TagStore.load() 从磁盘读取历史 tag，
TagStore.save() 是 seq 的唯一权威分配者（含文件锁防竞态）。

默认存储路径可通过 set_storage_dir() 配置。
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger

# 默认存储目录（可通过 set_storage_dir 覆盖）
_tags_dir: Path | None = None

MAX_RETRIES = 3
LOCK_TIMEOUT = 30  # 秒，超过此时间的锁视为过期（进程崩溃残留）


def set_storage_dir(path: str | Path) -> None:
    """设置 tag 文件存储目录。不调用则默认使用 cwd/logs/tags。"""
    global _tags_dir
    _tags_dir = Path(path)


def _get_tags_dir() -> Path:
    """返回 tag 存储目录（延迟初始化默认值）。"""
    global _tags_dir
    if _tags_dir is None:
        _tags_dir = Path.cwd() / "logs" / "tags"
    return _tags_dir


class TagStore:
    """Tag 持久化存储。

    load(): 读 JSON → list[dict]，文件不存在/损坏 → []
    save(): 原地写 tag_entry["seq"]，文件锁防竞态，损坏文件备份后重建。
    """

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _tag_path(session_id: str) -> Path:
        """构建 tag 文件路径，含路径穿越防护 + resolve() 验证。"""
        if ".." in session_id or "\\" in session_id:
            raise ValueError(f"session_id 含非法路径字符: {session_id!r}")
        safe = session_id.replace("/", "_").replace(":", "_").replace("\\", "_")
        path = _get_tags_dir() / f"{safe}.json"
        # 防御纵深: resolve() 后验证仍在 tags_dir 内
        resolved = path.resolve()
        tags_root = _get_tags_dir().resolve()
        try:
            resolved.relative_to(tags_root)
        except ValueError:
            raise ValueError(
                f"session_id 解析后路径超出允许范围: {session_id!r}"
            )
        return resolved

    @staticmethod
    def _acquire_lock(lock_path: Path) -> bool:
        """原子获取文件锁（O_CREAT|O_EXCL）。已获取或失败返回 False。

        过期锁（> LOCK_TIMEOUT）会被自动清理。
        """
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            return True
        except FileExistsError:
            # 检查是否过期
            try:
                mtime = lock_path.stat().st_mtime
                if time.time() - mtime > LOCK_TIMEOUT:
                    logger.warning(f"Tag 文件锁过期，强制清理: {lock_path}")
                    lock_path.unlink(missing_ok=True)
                    # 重试一次
                    try:
                        fd = os.open(
                            str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR
                        )
                        os.close(fd)
                        return True
                    except FileExistsError:
                        pass
            except OSError:
                pass
            return False
        except OSError:
            return False

    @staticmethod
    def _release_lock(lock_path: Path) -> None:
        """释放文件锁。"""
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _secure_perms(path: Path) -> None:
        """尽力把 tag 文件设为属主可读写(0o600)。

        tag 存的是对话 q/a(单用户数据, 非凭据)。best-effort 防纵深: POSIX 上限制
        为属主可读; Windows 上 chmod 仅影响只读位(近乎 no-op), 失败不阻塞。
        """
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @staticmethod
    def load(session_id: str) -> list[dict[str, Any]]:
        """加载 session 的所有 tag 记录。

        - 文件不存在 → 返回 []
        - JSON 损坏 → 记录警告 → 返回 []（不阻塞整个 session）
        """
        path = TagStore._tag_path(session_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(f"Tag 文件损坏，将返回空列表: {path}: {exc}")
            return []
        except OSError as exc:
            logger.warning(f"Tag 文件读取失败: {path}: {exc}")
            return []
        if not isinstance(data, list):
            logger.warning(f"Tag 文件格式异常（非列表）: {path}")
            return []
        return data

    @staticmethod
    def save(session_id: str, tag_entry: dict[str, Any]) -> None:
        """持久化一条 tag 记录，原地修改 tag_entry["seq"]。

        调用方传入不含 seq 的 dict，save() 内部计算并写入:
            tag_entry["seq"] = max_seq + 1

        文件锁防并发竞态。JSON 损坏 → 重命名备份 → 从头初始化。
        3 次重试均失败 → 记录错误日志 → 不抛异常（tag 丢失但不阻塞回复）。
        """
        path = TagStore._tag_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")

        for attempt in range(1, MAX_RETRIES + 1):
            # 获取锁
            if not TagStore._acquire_lock(lock_path):
                time.sleep(0.05 * attempt)  # 退避
                if attempt >= MAX_RETRIES:
                    logger.error(
                        f"TagStore.save 无法获取锁（{MAX_RETRIES}次尝试）: {lock_path}"
                    )
                    tag_entry["seq"] = -1  # 标记保存失败
                    return
                continue

            try:
                # ---- 临界区 ----
                existing: list[dict[str, Any]] = []
                try:
                    raw = (
                        path.read_text(encoding="utf-8")
                        if path.exists()
                        else "[]"
                    )
                    existing = json.loads(raw)
                    if not isinstance(existing, list):
                        raise json.JSONDecodeError("非列表格式", raw, 0)
                except (json.JSONDecodeError, TypeError) as exc:
                    ts = int(time.time())
                    backup_path = path.with_name(
                        f"{path.stem}.corrupted.{ts}{path.suffix}"
                    )
                    try:
                        path.rename(backup_path)
                        logger.warning(f"Tag 文件损坏，已备份: {backup_path}")
                    except OSError:
                        logger.warning(f"Tag 文件损坏，备份失败: {path}")
                    existing = []
                except OSError:
                    pass  # 文件不存在

                # 分配 seq
                max_seq = max(
                    (
                        t.get("seq", 0)
                        for t in existing
                        if isinstance(t, dict)
                    ),
                    default=0,
                )
                tag_entry["seq"] = max_seq + 1
                existing.append(tag_entry)

                # 写回
                path.write_text(
                    json.dumps(existing, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                TagStore._secure_perms(path)
                return  # 成功
                # ---- 临界区结束 ----
            except OSError as exc:
                if attempt >= MAX_RETRIES:
                    logger.error(
                        f"TagStore.save 写入失败（{MAX_RETRIES}次重试）: {path}: {exc}"
                    )
                    tag_entry["seq"] = -1  # 标记保存失败
                    return
                logger.warning(
                    f"TagStore.save 写入重试 {attempt}/{MAX_RETRIES}"
                )
            finally:
                TagStore._release_lock(lock_path)

    @staticmethod
    def delete(session_id: str, seqs: list[int]) -> int:
        """删除指定 seq 的 tag（编辑对话时清理被丢弃轮次），返回删除数量。

        与 save() 同一把文件锁，防与并发 save 竞态。删的是尾部高 seq；save()
        用 max_seq+1 续号，删后 max 降低、新轮次从此续号——被删轮次的前端消息
        同时被截断、无消息再引用这些 seq，故 seq 复用安全、不孤立保留消息的引用。
        """
        seq_set = {int(s) for s in seqs if isinstance(s, (int, float))}
        if not seq_set:
            return 0
        path = TagStore._tag_path(session_id)
        if not path.exists():
            return 0
        lock_path = path.with_suffix(path.suffix + ".lock")

        for attempt in range(1, MAX_RETRIES + 1):
            if not TagStore._acquire_lock(lock_path):
                time.sleep(0.05 * attempt)
                if attempt >= MAX_RETRIES:
                    logger.error(f"TagStore.delete 无法获取锁: {lock_path}")
                    return 0
                continue
            try:
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(existing, list):
                        return 0
                except (json.JSONDecodeError, TypeError, OSError):
                    return 0
                kept = [
                    t for t in existing
                    if not (isinstance(t, dict) and t.get("seq") in seq_set)
                ]
                removed = len(existing) - len(kept)
                if removed:
                    path.write_text(
                        json.dumps(kept, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    TagStore._secure_perms(path)
                return removed
            except OSError as exc:
                if attempt >= MAX_RETRIES:
                    logger.error(f"TagStore.delete 写入失败: {path}: {exc}")
                    return 0
                logger.warning(f"TagStore.delete 写入重试 {attempt}/{MAX_RETRIES}")
            finally:
                TagStore._release_lock(lock_path)
        return 0
