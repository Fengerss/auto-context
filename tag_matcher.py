"""Tag 三层上下文匹配器。

支持三种策略:
  - same_type: 维度 1+2（task_type + Jaccard），最快
  - llm:       维度 1+2+3（LLM 语义重排序），最准
  - embedding: 维度 1+2+3（Embedding 向量余弦），平衡

权重设计: semantic=3 > keywords=2 > task_type=1
语义相似度是上下文匹配的核心信号，task_type 仅作平局决胜。
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_SAME_TYPE_BONUS = 1
_JACCARD_WEIGHT = 2
_SEMANTIC_WEIGHT = 3


# ---------------------------------------------------------------------------
# MatchConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchConfig:
    """前端可传参覆盖的匹配策略。

    strategy:
      - same_type: 仅维度 1+2（task_type + Jaccard）— 最快，无外部调用
      - llm:       维度 1+2+3（LLM 语义重排序）— 最准，消耗 token
      - embedding: 维度 1+2+3（Embedding 向量余弦）— 平衡速度与语义

    max_qa_rounds: 最多注入几轮 QA 对。
      None = 全部候选都注入；0 = 零轮；正整数 = 最多 N 轮。

    llm_invoke: 异步 LLM 可调用对象，strategy="llm" 时必须传入。
    embed_fn: 批量 embedding 函数，strategy="embedding" 时必须传入。
    """

    strategy: Literal["same_type", "llm", "embedding"] = "same_type"
    max_qa_rounds: int | None = None
    llm_invoke: Callable[..., Any] | None = None
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None

    _ALLOWED: tuple[str, ...] = ("same_type", "llm", "embedding")

    def __post_init__(self):
        if self.strategy not in self._ALLOWED:
            raise ValueError(
                f"无效的匹配策略: {self.strategy!r}，允许: {self._ALLOWED}"
            )


# ---------------------------------------------------------------------------
# TagMatcher
# ---------------------------------------------------------------------------


class TagMatcher:
    """Tag 三层匹配器。

    可注入 llm_invoke / embed_fn（实例属性），便于测试:
        matcher = TagMatcher(llm_invoke=mock_llm)
    """

    def __init__(
        self,
        llm_invoke: Callable[..., Any] | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ):
        self._llm_invoke = llm_invoke
        self._embed_fn = embed_fn

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    async def match(
        self,
        keywords: list[str],
        tags: list[dict[str, Any]],
        task_type_hint: str,
        config: MatchConfig,
        query_text: str = "",
    ) -> list[int]:
        """按 keywords + task_type 匹配历史 tag，返回匹配的 seq 列表（score 降序）。

        query_text: 原始用户问题, LLM 语义匹配用它(比关键词串更能抓指代/改写)。

        Raises:
            ValueError: strategy 无效。
        """
        if not tags:
            return []

        # ---------- 维度 1+2: task_type + Jaccard ----------
        scored = self._score_candidates(keywords, tags, task_type_hint)

        # 过滤:
        #   same_type: 要求关键词重叠（score > task_type bonus），纯关键词匹配才有意义
        #   llm/embedding(语义): 不做关键词粗筛 —— 全部候选交语义层判相关(空关键词也可召回)
        if config.strategy == "same_type":
            candidates = [
                s for s in scored
                if s["_score"] > _SAME_TYPE_BONUS and isinstance(s.get("seq"), int)
            ]
        else:
            candidates = [s for s in scored if isinstance(s.get("seq"), int)]
        if not candidates:
            return []

        # ---------- 语义层(对全部候选, 不因数量<=1 跳过: 单条也让 LLM 判是否相关) ----------
        if config.strategy == "llm":
            candidates = await self._llm_rerank(keywords, candidates, config, query_text)
        elif config.strategy == "embedding":
            candidates = self._embedding_rerank(keywords, candidates, config)
        # same_type: 无语义层, 直接按关键词分排序

        # ---------- 排序 + top-K ----------
        candidates.sort(key=lambda c: c["_score"], reverse=True)
        if config.max_qa_rounds is not None and config.max_qa_rounds >= 0:
            candidates = candidates[: config.max_qa_rounds]

        return [c["seq"] for c in candidates]

    # ------------------------------------------------------------------
    # 维度 1: task_type 匹配（权重 1）+ 维度 2: Jaccard（权重 2）
    # ------------------------------------------------------------------

    @staticmethod
    def _score_candidates(
        keywords: list[str],
        tags: list[dict[str, Any]],
        task_type_hint: str,
    ) -> list[dict[str, Any]]:
        """对每个 tag 计算维度 1+2 得分，返回含 _score 的 tag 列表。"""
        kw_set = {k.lower() for k in keywords}
        scored: list[dict[str, Any]] = []

        for tag in tags:
            tag_type = tag.get("task_type", "")
            tag_kw = [
                k.lower()
                for k in tag.get("keywords", [])
                if isinstance(k, str)
            ]
            tag_kw_set = set(tag_kw)

            # 维度 1: task_type
            type_score = 0
            if tag_type == task_type_hint:
                type_score = _SAME_TYPE_BONUS

            # 维度 2: Jaccard
            intersection = kw_set & tag_kw_set
            union = kw_set | tag_kw_set
            jaccard = len(intersection) / len(union) if union else 0.0

            total = type_score + jaccard * _JACCARD_WEIGHT
            scored.append({**tag, "_score": total})

        return scored

    # ------------------------------------------------------------------
    # 维度 3a: LLM 语义重排序
    # ------------------------------------------------------------------

    async def _llm_rerank(
        self,
        keywords: list[str],
        candidates: list[dict[str, Any]],
        config: MatchConfig,
        query_text: str = "",
    ) -> list[dict[str, Any]]:
        """调用 LLM 对候选 tag 列表做相关性选择+排序。

        去粗筛后 candidates=全部 tag, LLM 负责判相关。兜底语义(关键):
        - LLM 不可用/异常 → 退回关键词过滤(score>bonus), 不注全部
        - LLM 明确返回空(判无相关) → 注空, 尊重 LLM(否则去粗筛后会把全部历史都注进去)
        """
        invoke = config.llm_invoke or self._llm_invoke
        # 关键词兜底: 只保留有关键词重叠的(等价 same_type), 避免无 LLM 时注入全部
        kw_fallback = [c for c in candidates if c["_score"] > _SAME_TYPE_BONUS]
        if invoke is None:
            return kw_fallback

        query = query_text.strip() or " ".join(keywords)
        tag_lines = [
            f'seq={t["seq"]} title="{t.get("title", "")}" '
            f'keywords={t.get("keywords", [])}'
            for t in candidates
        ]
        tag_list_text = "\n".join(tag_lines)

        prompt = f"""你是上下文选择器。判断当前用户问题与哪些历史对话标签真正相关。
只选真正相关的(能为当前问题提供上下文的), 按相关度降序返回其 seq。都不相关就返回空数组。

当前问题: {query}

历史标签:
{tag_list_text}

只返回 JSON 数组，不加解释:
[1, 3]"""

        try:
            response = await invoke(prompt)
            content = getattr(response, "content", str(response))
            ordered_seqs = TagMatcher._parse_llm_seqs(content)
        except Exception:
            return kw_fallback  # LLM 异常 → 关键词兜底(非注全部)

        # 过滤有效 seq
        valid_seqs = {t["seq"] for t in candidates}
        ordered_seqs = [s for s in ordered_seqs if s in valid_seqs]

        if not ordered_seqs:
            return []  # LLM 判无相关 → 注空(不能回退注全部候选)

        # LLM 返回的排序覆盖原始 score 排序
        seq_to_candidate = {c["seq"]: c for c in candidates}
        reranked = []
        for i, seq in enumerate(ordered_seqs):
            c = dict(seq_to_candidate[seq])
            c["_score"] = len(ordered_seqs) - i  # 越靠前分越高
            reranked.append(c)
        return reranked

    @staticmethod
    def _parse_llm_seqs(content: str) -> list[int]:
        """容错解析 LLM 返回的 seq 列表。

        每字段独立守卫，单个字段损坏不影响解析。
        """
        try:
            parsed = json.loads(content.strip())
        except (json.JSONDecodeError, TypeError):
            return []

        if isinstance(parsed, list):
            return [
                int(s) for s in parsed
                if isinstance(s, (int, float)) and s == s  # NaN guard
            ]

        if isinstance(parsed, dict):
            ordered = parsed.get("seqs") or parsed.get("relevant") or []
            if isinstance(ordered, list):
                return [
                    int(s) for s in ordered
                    if isinstance(s, (int, float)) and s == s
                ]

        return []

    # ------------------------------------------------------------------
    # 维度 3b: Embedding 向量余弦
    # ------------------------------------------------------------------

    def _embedding_rerank(
        self,
        keywords: list[str],
        candidates: list[dict[str, Any]],
        config: MatchConfig,
    ) -> list[dict[str, Any]]:
        """Embedding 余弦相似度叠加到总分，返回重排序的 candidates。"""
        embed_fn = config.embed_fn or self._embed_fn
        if embed_fn is None:
            return candidates  # 无 embed_fn → 回退

        query_text = " ".join(keywords)
        tag_texts = [
            f"{c.get('title', '')} {' '.join(c.get('keywords', []))}"
            for c in candidates
        ]

        try:
            embeddings = embed_fn([query_text] + tag_texts)
        except Exception:
            return candidates  # embedding 失败 → 回退

        if not embeddings or len(embeddings) < 2:
            return candidates

        query_emb = embeddings[0]
        for i, tag_emb in enumerate(embeddings[1:]):
            cosine = TagMatcher._cosine_similarity(query_emb, tag_emb)
            candidates[i]["_score"] += cosine * _SEMANTIC_WEIGHT

        return candidates

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """两个向量的余弦相似度。"""
        if len(a) != len(b) or len(a) == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
