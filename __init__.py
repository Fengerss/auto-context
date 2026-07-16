"""Tag 系统 — 对话上下文持久化与三层匹配引擎。

从 langGraph_agent/data_agent/chatbi_graph 拆分出来的独立模块。

核心组件:
- TagStore: JSON 文件持久化，文件锁防竞态，按 session_id 隔离
- TagMatcher: 三层匹配 (same_type/Jaccard → LLM 语义 → Embedding 余弦)
- TagBuilder: 图后 tag 生成 (关键词提取、数据源记录、上下文构建)

使用示例:
    from tag_system import TagStore, TagMatcher, MatchConfig, build_tag_entry

    # 加载历史 tag
    tags = TagStore.load(session_id)

    # 匹配相关上下文
    matcher = TagMatcher(llm_invoke=llm.ainvoke)
    cfg = MatchConfig(strategy="llm", max_qa_rounds=5)
    matched = await matcher.match(keywords, tags, "data_analysis", cfg, query_text="查询")

    # 保存新 tag
    entry = build_tag_entry(observations, answer, query, "data_analysis")
    TagStore.save(session_id, entry)
"""

from tag_system.tag_store import TagStore
from tag_system.tag_matcher import MatchConfig, TagMatcher
from tag_system.tag_builder import (
    TagBuilder,
    build_tag_entry,
    build_match_config,
    build_context_from_tags,
    tokenize,
    condense_answer,
    SRC_LABEL,
)

__all__ = [
    "TagStore",
    "MatchConfig",
    "TagMatcher",
    "TagBuilder",
    "build_tag_entry",
    "build_match_config",
    "build_context_from_tags",
    "tokenize",
    "condense_answer",
    "SRC_LABEL",
]
