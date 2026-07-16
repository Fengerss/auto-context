"""Tag 构建器 — 图后 tag 生成 + 上下文构建 + 分词 + 匹配配置。

从 chat_api.py 中提取的 tag 相关工具函数，独立于 FastAPI/图执行。
"""

import datetime as _dt
import re
from typing import Any

import jieba

# 数据源 code → 中文来源标签
SRC_LABEL: dict[str, str] = {
    "db": "业务数据库",
    "sqlite": "上传表格(本地表)",
    "kb": "知识库",
}


def tokenize(text: str, limit: int = 15) -> list[str]:
    """jieba 中文分词 + 英文词, 供 Tag 关键词 Jaccard 匹配。

    整串中文当一个 token 会让 Jaccard≈0, jieba 切成词后相关问句才有重叠。
    """
    seen: list[str] = []
    for tok in jieba.cut(text or ""):
        t = tok.strip()
        if len(t) >= 2 and t not in seen:
            seen.append(t)
    return seen[:limit]


def condense_answer(text: str) -> str:
    """压缩历史答案供注入: 剥图片语法 + Markdown 表格行。

    表格/图片是注入噪声(gen 写 SQL 用不上), 且旧图链接会诱导模型重放过期图片。
    保留正文与要点数字。
    """
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text or "")  # 图片语法
    kept = [
        ln for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("|")  # 丢表格行/分隔行
    ]
    return "\n".join(kept).strip()


def build_tag_entry(
    observations: list[dict],
    final_answer: str,
    user_query: str = "",
    task_type: str = "",
    keywords: list[str] | None = None,
    title: str = "",
) -> dict[str, Any]:
    """图后生成 TagEntry。从 observations 提取 data_source 和 tools_used。

    data_source 为数组 — 记录所有涉及的源: ["db", "sqlite", "kb"]。
    task_type: 存下来让 TagMatcher 的 same_type 维度生效。
    keywords/title: 优先用意图分类器规范化的关键词/主题标签(整词/去噪/概括); 无则回退。
    """
    data_sources: list[str] = []
    tools_used: list[str] = []
    for obs in observations or []:
        src = obs.get("source")
        if src and src not in data_sources:
            data_sources.append(src)
        tn = obs.get("tool_name")
        if tn and tn not in tools_used:
            tools_used.append(tn)

    # title: 优先 LLM 规范主题标签; 回退到 query 截断(旧行为)
    tag_title = (title.strip() if title else (user_query or final_answer or "查询")[:20]).strip()

    # keywords: 优先 LLM(整词, 避免 jieba 把"王一珂"切成"王一"); 回退 jieba
    kw = list(keywords) if keywords else (tokenize(user_query, limit=8) if user_query else [])

    return {
        "task_type": task_type,
        "data_source": data_sources,
        "tools_used": tools_used,
        "title": tag_title,
        "keywords": kw,
        "q": user_query or "",       # 存全文(注入侧再压缩), 上下文注入不依赖前端 history
        "a": final_answer or "",
        "timestamp": _dt.datetime.now().isoformat(),
    }


def build_context_from_tags(
    matched_seqs: list[int],
    tags: list[dict],
    src_label: dict[str, str] | None = None,
) -> str:
    """从匹配到的 tag 自带的 q/a 直接构建跨轮上下文, 不依赖前端 history 布局。

    旧格式 tag 无 q/a → 退用 title 作 q, 仍不碰前端 history。
    q/a 存的是全文, 注入时压缩(剥图表)并按预算截断。
    每轮标注 tag 的 data_source(来源) —— 否则 gen_sql/answer 看到的是无源散文,
    会把 sqlite 的"892 行"当 db 数据用(跨源污染 → 误路由/编造)。
    """
    labels = src_label or SRC_LABEL
    by_seq = {t.get("seq"): t for t in tags if isinstance(t, dict)}
    lines = []
    for seq in matched_seqs:
        t = by_seq.get(seq)
        if not t:
            continue
        q = (t.get("q") or t.get("title") or "").strip()
        a = condense_answer(t.get("a") or "")
        srcs = t.get("data_source") or []
        src_label_str = "、".join(labels.get(s, s) for s in srcs if s)
        tag_mark = f"[来源:{src_label_str}]" if src_label_str else ""
        if q:
            lines.append(f"用户: {q[:300]}")
        if a:
            lines.append(f"助手{tag_mark}: {a[:2000]}")
    return "\n".join(lines)


def build_match_config(raw: dict) -> "MatchConfig":
    """从前端 match_config dict 构建 MatchConfig，含白名单 + 类型校验。

    Raises:
        ValueError: strategy/max_qa_rounds 无效。
    """
    from tag_system.tag_matcher import MatchConfig

    strategy = raw.get("strategy", "llm")
    if strategy not in ("same_type", "llm", "embedding"):
        raise ValueError(f"无效的 match_config.strategy: {strategy!r}")

    max_qa = raw.get("max_qa_rounds", 5)
    if max_qa is not None:
        if isinstance(max_qa, bool) or not isinstance(max_qa, int):
            raise ValueError(
                f"max_qa_rounds 必须是整数，得到 {type(max_qa).__name__}: {max_qa!r}"
            )
        if max_qa < 0:
            raise ValueError(f"max_qa_rounds 不能为负数: {max_qa}")

    return MatchConfig(
        strategy=strategy,
        max_qa_rounds=max_qa,
    )


class TagBuilder:
    """Tag 构建器（便于依赖注入测试）。

    用法:
        builder = TagBuilder()
        entry = builder.build(observations, answer, query, "data_analysis")
        TagStore.save(session_id, entry)
    """

    def build(
        self,
        observations: list[dict],
        final_answer: str,
        user_query: str = "",
        task_type: str = "",
        keywords: list[str] | None = None,
        title: str = "",
    ) -> dict[str, Any]:
        return build_tag_entry(observations, final_answer, user_query, task_type, keywords, title)

    def build_context(
        self,
        matched_seqs: list[int],
        tags: list[dict],
        src_label: dict[str, str] | None = None,
    ) -> str:
        return build_context_from_tags(matched_seqs, tags, src_label)

    @staticmethod
    def tokenize(text: str, limit: int = 15) -> list[str]:
        return tokenize(text, limit)
