"""TagStore 持久化单元测试。"""
import pytest

import tag_system.tag_store as tag_store_mod
from tag_system.tag_store import TagStore, set_storage_dir


class TestTagStore:
    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tag_store_mod, "_tags_dir", tmp_path / "tags")

        entry = {"title": "测试", "keywords": ["test"]}
        TagStore.save("test-session-1", entry)
        assert entry.get("seq") == 1

        loaded = TagStore.load("test-session-1")
        assert len(loaded) == 1
        assert loaded[0]["title"] == "测试"
        assert loaded[0]["seq"] == 1

    def test_multiple_saves_sequence(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tag_store_mod, "_tags_dir", tmp_path / "tags")

        for i in range(3):
            entry = {"title": f"test-{i}"}
            TagStore.save("test-session-2", entry)
            assert entry["seq"] == i + 1

        loaded = TagStore.load("test-session-2")
        assert len(loaded) == 3
        assert [t["seq"] for t in loaded] == [1, 2, 3]

    def test_load_empty_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tag_store_mod, "_tags_dir", tmp_path / "tags")
        loaded = TagStore.load("nonexistent-session")
        assert loaded == []

    def test_corrupt_json_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tag_store_mod, "_tags_dir", tmp_path / "tags")

        path = TagStore._tag_path("corrupt-session")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("this is not json{{{", encoding="utf-8")

        loaded = TagStore.load("corrupt-session")
        assert loaded == []

    def test_save_after_corrupt(self, tmp_path, monkeypatch):
        """损坏文件被覆盖后, save 仍能成功"""
        monkeypatch.setattr(tag_store_mod, "_tags_dir", tmp_path / "tags")

        path = TagStore._tag_path("recover-session")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("broken{{{", encoding="utf-8")

        entry = {"title": "new"}
        TagStore.save("recover-session", entry)
        assert entry.get("seq") is not None

    def test_acquire_lock_failure_non_blocking(self, tmp_path, monkeypatch):
        """锁获取失败时不抛异常, tag 标记 seq=-1"""
        monkeypatch.setattr(tag_store_mod, "_tags_dir", tmp_path / "tags")
        monkeypatch.setattr(TagStore, "_acquire_lock", staticmethod(lambda _p: False))

        entry = {"title": "lock-fail"}
        TagStore.save("lock-fail-test", entry)
        assert entry.get("seq") == -1

    def test_delete_removes_tail_and_seq_continues(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tag_store_mod, "_tags_dir", tmp_path / "tags")
        sid = "edit-session"
        for i in range(5):
            TagStore.save(sid, {"title": f"t{i}"})
        assert [t["seq"] for t in TagStore.load(sid)] == [1, 2, 3, 4, 5]

        removed = TagStore.delete(sid, [3, 4, 5])
        assert removed == 3
        assert [t["seq"] for t in TagStore.load(sid)] == [1, 2]

        entry = {"title": "rerun"}
        TagStore.save(sid, entry)
        assert entry["seq"] == 3
        assert [t["seq"] for t in TagStore.load(sid)] == [1, 2, 3]

    def test_delete_noop_cases(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tag_store_mod, "_tags_dir", tmp_path / "tags")
        sid = "noop-session"
        TagStore.save(sid, {"title": "a"})
        assert TagStore.delete(sid, []) == 0
        assert TagStore.delete(sid, [99]) == 0
        assert TagStore.delete("no-such-session", [1]) == 0
        assert len(TagStore.load(sid)) == 1

    def test_session_id_safety(self):
        """路径穿越防护。"""
        with pytest.raises(ValueError):
            TagStore._tag_path("../etc/passwd")
        with pytest.raises(ValueError):
            TagStore._tag_path("a\\b")

    def test_set_storage_dir(self, tmp_path):
        """set_storage_dir 后 TagStore 使用新路径。"""
        set_storage_dir(str(tmp_path / "custom_tags"))
        try:
            entry = {"title": "custom-path"}
            TagStore.save("custom-session", entry)
            assert entry.get("seq") == 1
            assert (tmp_path / "custom_tags" / "custom-session.json").exists()
        finally:
            # 重置为默认（避免影响其他测试）
            set_storage_dir.__wrapped__(None) if hasattr(set_storage_dir, "__wrapped__") else None
            import tag_system.tag_store as m
            m._tags_dir = None
