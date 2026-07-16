"""TagMatcher 三层匹配单元测试。"""
import asyncio
import pytest

from tag_system.tag_matcher import TagMatcher, MatchConfig


class TestTagMatcher:
    def test_same_type_match(self):
        matcher = TagMatcher()
        tags = [
            {"seq": 1, "task_type": "data_analysis", "keywords": ["销售", "数据"]},
            {"seq": 2, "task_type": "free_chat", "keywords": ["你好"]},
        ]
        cfg = MatchConfig(strategy="same_type")
        result = asyncio.run(
            matcher.match(["销售", "分析"], tags, "data_analysis", cfg)
        )
        assert 1 in result

    def test_same_type_no_match(self):
        matcher = TagMatcher()
        tags = [
            {"seq": 1, "task_type": "free_chat", "keywords": ["你好"]},
        ]
        cfg = MatchConfig(strategy="same_type")
        result = asyncio.run(
            matcher.match(["销售"], tags, "data_analysis", cfg)
        )
        assert result == []

    def test_score_candidates_jaccard_order(self):
        matcher = TagMatcher()
        tags = [
            {"seq": 1, "task_type": "data_analysis", "keywords": ["销售", "食品"]},
            {"seq": 2, "task_type": "data_analysis", "keywords": ["销售", "数据", "趋势"]},
        ]
        candidates = matcher._score_candidates(["销售", "数据"], tags, "data_analysis")
        assert len(candidates) == 2
        seqs = {c["seq"] for c in candidates}
        assert seqs == {1, 2}

    def test_llm_strategy_no_crash(self):
        """LLM 策略执行不崩溃"""
        async def mock_llm(*args, **kwargs):
            return type("R", (), {"content": '{"relevant_seqs": [1]}'})()

        matcher = TagMatcher(llm_invoke=mock_llm)
        tags = [
            {"seq": 1, "task_type": "data_analysis", "keywords": ["销售", "数据"]},
            {"seq": 2, "task_type": "data_analysis", "keywords": ["销售", "趋势"]},
        ]
        cfg = MatchConfig(strategy="llm", llm_invoke=mock_llm)
        result = asyncio.run(matcher.match(["销售"], tags, "data_analysis", cfg))
        assert isinstance(result, list)

    def test_llm_strategy_fallback_on_error(self):
        async def mock_llm_fail(*args, **kwargs):
            raise RuntimeError("LLM unavailable")

        matcher = TagMatcher(llm_invoke=mock_llm_fail)
        tags = [
            {"seq": 1, "task_type": "data_analysis", "keywords": ["销售"]},
        ]
        cfg = MatchConfig(strategy="llm", llm_invoke=mock_llm_fail)
        result = asyncio.run(matcher.match(["销售"], tags, "data_analysis", cfg))
        assert isinstance(result, list)

    def test_embedding_strategy_fallback_on_none(self):
        matcher = TagMatcher(embed_fn=lambda texts: None)
        tags = [
            {"seq": 1, "task_type": "data_analysis", "keywords": ["销售"]},
        ]
        cfg = MatchConfig(strategy="embedding", embed_fn=lambda texts: None)
        result = asyncio.run(matcher.match(["销售"], tags, "data_analysis", cfg))
        assert isinstance(result, list)

    def test_embedding_strategy_cosine_order(self):
        def mock_embed(texts):
            out = []
            for t in texts:
                if "销售" in t:
                    out.append([1.0, 0.0])
                elif "食品" in t:
                    out.append([0.9, 0.1])
                else:
                    out.append([0.0, 1.0])
            return out

        matcher = TagMatcher(embed_fn=mock_embed)
        tags = [
            {"seq": 1, "task_type": "data_analysis", "keywords": ["销售", "数据"]},
            {"seq": 2, "task_type": "data_analysis", "keywords": ["食品", "价格"]},
        ]
        cfg = MatchConfig(strategy="embedding", embed_fn=mock_embed)
        result = asyncio.run(matcher.match(["销售"], tags, "data_analysis", cfg))
        assert len(result) >= 1

    def test_none_task_type_hint(self):
        """task_type=None 时兼容新旧 tag 格式混存, 不崩溃"""
        matcher = TagMatcher()
        tags = [
            {"seq": 1, "task_type": "data_analysis", "keywords": ["销售"]},
            {"seq": 2, "data_source": ["db"], "keywords": ["用户"]},
        ]
        cfg = MatchConfig(strategy="embedding", embed_fn=lambda texts: None)
        result = asyncio.run(matcher.match(["销售"], tags, None, cfg))
        assert isinstance(result, list)

    def test_llm_no_prefilter_reaches_keyword_miss(self):
        """语义策略去粗筛: 关键词零重叠的 tag, LLM 选中也能返回"""
        async def mock_pick2(*a, **k):
            return type("R", (), {"content": "[2]"})()

        matcher = TagMatcher(llm_invoke=mock_pick2)
        tags = [
            {"seq": 1, "task_type": "data_analysis", "keywords": ["销售"]},
            {"seq": 2, "task_type": "rag", "keywords": ["退货", "规定"]},
        ]
        cfg = MatchConfig(strategy="llm", llm_invoke=mock_pick2)
        result = asyncio.run(
            matcher.match(["销售"], tags, "data_analysis", cfg, query_text="上次那个退货怎么说的")
        )
        assert result == [2]

    def test_llm_empty_result_injects_nothing(self):
        """LLM 判无相关(返回 [])→ 注空, 不能回退注全部候选"""
        async def mock_empty(*a, **k):
            return type("R", (), {"content": "[]"})()

        matcher = TagMatcher(llm_invoke=mock_empty)
        tags = [
            {"seq": 1, "task_type": "data_analysis", "keywords": ["销售"]},
            {"seq": 2, "task_type": "data_analysis", "keywords": ["天气"]},
        ]
        cfg = MatchConfig(strategy="llm", llm_invoke=mock_empty)
        result = asyncio.run(
            matcher.match(["随便"], tags, "data_analysis", cfg, query_text="随便问问")
        )
        assert result == []

    def test_llm_error_keyword_fallback_not_all(self):
        """LLM 异常 → 退回关键词过滤, 不注全部候选"""
        async def mock_fail(*a, **k):
            raise RuntimeError("down")

        matcher = TagMatcher(llm_invoke=mock_fail)
        tags = [
            {"seq": 1, "task_type": "data_analysis", "keywords": ["销售", "数据"]},
            {"seq": 2, "task_type": "rag", "keywords": ["天气"]},
        ]
        cfg = MatchConfig(strategy="llm", llm_invoke=mock_fail)
        result = asyncio.run(
            matcher.match(["销售"], tags, "data_analysis", cfg, query_text="销售")
        )
        assert result == [1]

    def test_max_qa_rounds_limit(self):
        """max_qa_rounds 截断结果"""
        matcher = TagMatcher()
        tags = [
            {"seq": i, "task_type": "data_analysis", "keywords": ["销售", f"k{i}"]}
            for i in range(1, 6)
        ]
        cfg = MatchConfig(strategy="same_type", max_qa_rounds=2)
        result = asyncio.run(matcher.match(["销售"], tags, "data_analysis", cfg))
        assert len(result) <= 2

    def test_strategy_validation(self):
        """无效 strategy 抛 ValueError"""
        with pytest.raises(ValueError):
            MatchConfig(strategy="invalid_strategy")  # type: ignore[arg-type]

    def test_cosine_similarity(self):
        """余弦相似度计算正确"""
        assert TagMatcher._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
        assert TagMatcher._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
        assert TagMatcher._cosine_similarity([], [1.0]) == 0.0
