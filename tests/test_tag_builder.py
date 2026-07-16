"""TagBuilder 单元测试 — tag 生成 + 分词 + 上下文构建。"""
from tag_system.tag_builder import (
    build_tag_entry,
    tokenize,
    condense_answer,
    build_context_from_tags,
    build_match_config,
)


class TestTokenize:
    def test_chinese_tokenization(self):
        tokens = tokenize("查询运动类商品有多少")
        assert len(tokens) >= 1  # jieba 至少切出一个词

    def test_empty_input(self):
        assert tokenize("") == []

    def test_deduplication(self):
        tokens = tokenize("查询查询查询")
        assert len(tokens) <= len(set(tokens))  # 无重复

    def test_limit(self):
        # "销售 数据 用户 画像 分析 趋势" 等, limit=3 只取前 3
        tokens = tokenize("销售 数据 用户 画像 分析 趋势 预测 图表", limit=3)
        assert len(tokens) <= 3


class TestCondenseAnswer:
    def test_removes_image_syntax(self):
        text = "这是结果 ![图表](figure_abc123.png) 分析完成"
        result = condense_answer(text)
        assert "![图表]" not in result
        assert "分析完成" in result

    def test_removes_table_rows(self):
        text = "结果如下\n| 列1 | 列2 |\n| 100 | 200 |\n分析完成"
        result = condense_answer(text)
        assert "| 列1" not in result
        assert "分析完成" in result

    def test_empty_input(self):
        assert condense_answer("") == ""
        assert condense_answer(None) == ""


class TestBuildTagEntry:
    def test_data_source_db(self):
        entry = build_tag_entry(
            [{"kind": "sql", "source": "db", "tool_name": "db_sql_tool", "success": True}],
            "查询结果", "列出所有表",
        )
        assert entry["data_source"] == ["db"]

    def test_data_source_multi(self):
        entry = build_tag_entry(
            [
                {"kind": "sql", "source": "db", "tool_name": "db_sql_tool", "success": True},
                {"kind": "sql", "source": "sqlite", "tool_name": "sqlite_query", "success": True},
            ],
            "对比结果", "对比数据",
        )
        assert "db" in entry["data_source"]
        assert "sqlite" in entry["data_source"]

    def test_data_source_analysis(self):
        entry = build_tag_entry(
            [{"kind": "rag", "source": "kb", "tool_name": "search_knowledge_base_tool", "success": True}],
            "检索完成", "查退货规定",
        )
        assert "kb" in entry["data_source"]

    def test_keywords_extraction(self):
        entry = build_tag_entry(
            [], "结果", "sales data user profile 分析",
        )
        assert len(entry["keywords"]) >= 2

    def test_tools_used_from_observations(self):
        entry = build_tag_entry(
            [
                {"kind": "sql", "source": "db", "tool_name": "db_sql_tool", "success": True},
                {"kind": "analysis", "source": None, "tool_name": "run_python_script_tool", "success": True},
            ],
            "结果", "查询",
        )
        assert "db_sql_tool" in entry["tools_used"]
        assert "run_python_script_tool" in entry["tools_used"]

    def test_empty_history_no_crash(self):
        """无 action 时 tag 不崩溃 (纯闲聊)"""
        entry = build_tag_entry([], "闲聊回复", "")
        assert entry["data_source"] == []
        assert entry["tools_used"] == []
        assert isinstance(entry["title"], str)

    def test_llm_keywords_priority(self):
        """LLM keywords 优先于 jieba 分词"""
        entry = build_tag_entry(
            [], "结果", "查询王一珂的订单",
            keywords=["王一珂", "订单"],
        )
        assert "王一珂" in entry["keywords"]
        assert "订单" in entry["keywords"]

    def test_timestamp_present(self):
        entry = build_tag_entry([], "结果", "查询")
        assert "timestamp" in entry
        assert "T" in entry["timestamp"]  # ISO 格式


class TestBuildContextFromTags:
    def test_basic_context(self):
        tags = [
            {"seq": 1, "title": "运动商品", "q": "查询运动类商品", "a": "共 100 种", "data_source": ["db"]},
        ]
        ctx = build_context_from_tags([1], tags)
        assert "查询运动类商品" in ctx
        assert "共 100 种" in ctx
        assert "业务数据库" in ctx

    def test_missing_seq(self):
        ctx = build_context_from_tags([99], [])
        assert ctx == ""

    def test_old_format_fallback(self):
        """旧格式无 q/a → 用 title"""
        tags = [
            {"seq": 1, "title": "运动商品统计"},
        ]
        ctx = build_context_from_tags([1], tags)
        assert "运动商品统计" in ctx

    def test_custom_src_label(self):
        tags = [
            {"seq": 1, "q": "查询", "a": "结果", "data_source": ["db"]},
        ]
        custom = {"db": "自定义数据库"}
        ctx = build_context_from_tags([1], tags, src_label=custom)
        assert "自定义数据库" in ctx


class TestBuildMatchConfig:
    def test_defaults(self):
        cfg = build_match_config({})
        assert cfg.strategy == "llm"
        assert cfg.max_qa_rounds == 5

    def test_custom_values(self):
        cfg = build_match_config({"strategy": "same_type", "max_qa_rounds": 3})
        assert cfg.strategy == "same_type"
        assert cfg.max_qa_rounds == 3

    def test_invalid_strategy(self):
        import pytest
        with pytest.raises(ValueError, match="无效的.*strategy"):
            build_match_config({"strategy": "invalid"})

    def test_negative_max_qa(self):
        import pytest
        with pytest.raises(ValueError, match="不能为负数"):
            build_match_config({"max_qa_rounds": -1})
