# auto-context — LLM 自主上下文管理 <sup>v0.1.0</sup>

让 LLM 自己管理对话上下文，不再需要手动裁剪历史、拼接 prompt。

## 问题

长对话中，LLM 的上下文窗口有限。传统做法是：

- 手动挑选"相关"的历史轮次拼进 prompt
- 按固定窗口截断（最近 N 轮），旧信息直接丢失
- 依赖前端 `history` 数组，跨 session 无法复用

这些都是**人在管理上下文**——判断什么相关、什么该丢、什么该留。

## 方案

**Tag System 让 LLM 自己管理上下文。** 全流程闭环：

```
对话结束 → LLM 生成 tag（标题 + 关键词）→ 持久化
新对话   → LLM 语义匹配相关 tag      → 注入上下文
```

- **打 tag 的是 LLM**：每轮对话后，LLM 自动提取规范化的标题和关键词（整词、去噪、概括），不依赖分词器切词
- **匹配的也是 LLM**：新问题来了，LLM 从历史 tag 中选出真正相关的，理解指代和改写，不靠关键词重叠碰运气

jieba 分词仅作为 LLM 关键词不可用时的回退方案。

## 核心组件

### TagStore — 持久化存储

- JSON 文件按 `session_id` 隔离
- 文件锁（`O_CREAT|O_EXCL`）防并发竞态
- 损坏文件自动备份重建，不阻塞会话

```python
from tag_system import TagStore

tags = TagStore.load("session_abc")     # 加载历史 tag
TagStore.save("session_abc", entry)     # 保存新 tag，自动分配 seq
```

### TagBuilder — LLM 自动 tag 生成

每轮对话后，由 LLM 生成 tag 的核心字段：

- **title**: LLM 提取的规范化主题标签（如"Q3销售额分析"），概括本轮内容
- **keywords**: LLM 提取的整词关键词（如 `["销售额", "Q3", "同比"]`），避免分词器切碎专有名词
- **data_source / tools_used**: 从 observations 自动提取，记录本轮涉及的数据源和工具

LLM 不可用时回退到 jieba 分词 + query 截断。

```python
from tag_system import build_tag_entry, build_context_from_tags

# LLM 生成 title 和 keywords 后传入
entry = build_tag_entry(
    observations, answer, query, "data_analysis",
    title="Q3销售额同比分析",           # ← LLM 生成的标题
    keywords=["销售额", "Q3", "同比"],  # ← LLM 生成的关键词
)
context_text = build_context_from_tags(matched_seqs, tags)
```

### Tag 结构

每条 tag 是一个 JSON 对象，由 `build_tag_entry()` 生成、`TagStore.save()` 写入时自动分配 `seq`：

```json
{
  "seq": 1,
  "task_type": "data_analysis",
  "data_source": ["db", "kb"],
  "tools_used": ["sql_query"],
  "title": "Q3销售额同比分析",
  "keywords": ["销售额", "Q3", "同比"],
  "q": "Q3 销售额同比变化多少？",
  "a": "Q3 销售额同比增长 15%，环比增长 3%...",
  "timestamp": "2026-07-16T21:45:00"
}
```

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `seq` | `int` | `TagStore.save()` 自动分配 | 单调递增序号，匹配结果按此排序 |
| `task_type` | `str` | 调用方传入 | 任务类型，供 `same_type` 维度匹配 |
| `data_source` | `list[str]` | observations 自动提取 | 涉及的数据源：`db` / `sqlite` / `kb` |
| `tools_used` | `list[str]` | observations 自动提取 | 使用的工具名称 |
| `title` | `str` | **LLM 生成**（回退：query 截断） | 规范化主题标签，概括本轮内容 |
| `keywords` | `list[str]` | **LLM 生成**（回退：jieba 分词） | 整词关键词，用于 Jaccard 初筛 |
| `q` | `str` | 调用方传入 | 用户原始问题全文 |
| `a` | `str` | 调用方传入 | LLM 回复全文（注入时压缩） |
| `timestamp` | `str` | 自动生成 | ISO 8601 时间戳 |

### TagMatcher — 三层匹配引擎

| 策略 | 维度 | 速度 | 精度 | 适用场景 |
|------|------|------|------|----------|
| `same_type` | task_type + Jaccard | 最快 | 基础 | 关键词明确、无需语义理解 |
| `llm` | 以上 + LLM 语义重排序 | 较慢 | 最准 | 需要理解指代、改写、隐含意图 |
| `embedding` | 以上 + Embedding 余弦 | 中等 | 较高 | 平衡速度与语义，无需消耗 LLM token |

权重设计：**semantic(3) > keywords(2) > task_type(1)**
语义相似度是核心信号，task_type 仅作平局决胜。

```python
from tag_system import TagMatcher, MatchConfig

matcher = TagMatcher(llm_invoke=llm.ainvoke)
cfg = MatchConfig(strategy="llm", max_qa_rounds=5)
matched = await matcher.match(keywords, tags, "data_analysis", cfg, query_text=query)
```

## 安装

```bash
pip install jieba loguru
```

- `loguru` — 日志
- `jieba` — 中文分词（仅 LLM 关键词不可用时的回退方案）

## 快速开始

```python
from tag_system import TagStore, TagMatcher, MatchConfig, build_tag_entry, build_context_from_tags

session_id = "user_123"
query = "Q3 销售额同比变化"

# 1. 加载历史 tag
tags = TagStore.load(session_id)

# 2. 先让 LLM 从当前问题提取关键词和标题（这一步在匹配之前）
llm_keywords = await extract_keywords_via_llm(query)  # → ["销售额", "Q3", "同比"]
llm_title = await extract_title_via_llm(query)        # → "Q3销售额同比分析"

# 3. 用 LLM 提取的关键词匹配历史上下文
matcher = TagMatcher(llm_invoke=your_llm_invoke)
cfg = MatchConfig(strategy="llm", max_qa_rounds=5)
matched_seqs = await matcher.match(
    keywords=llm_keywords,
    tags=tags,
    task_type_hint="data_analysis",
    config=cfg,
    query_text=query,
)

# 4. 注入匹配到的上下文
context = build_context_from_tags(matched_seqs, tags)
prompt = f"历史相关上下文:\n{context}\n\n当前问题: {query}"

# 5. LLM 回复后，保存本轮 tag（title 和 keywords 由 LLM 生成）
entry = build_tag_entry(
    observations=[...],
    final_answer="Q3 销售额同比增长 15%...",
    user_query=query,
    task_type="data_analysis",
    title=llm_title,           # ← LLM 生成的标题
    keywords=llm_keywords,     # ← LLM 生成的关键词
)
TagStore.save(session_id, entry)
```

## 容错设计

| 场景 | 行为 |
|------|------|
| Tag 文件不存在 | 返回空列表，不阻塞 |
| JSON 损坏 | 备份 → 重建，记录警告 |
| LLM 不可用 | 退回 Jaccard 关键词匹配 |
| Embedding 失败 | 退回关键词匹配 |
| 文件锁超时(30s) | 强制清理过期锁 |

## 目录结构

```
tag_system/
├── __init__.py          # 公开 API
├── tag_store.py         # TagStore: 持久化 + 文件锁
├── tag_builder.py       # TagBuilder: tag 生成 + 分词 + 上下文构建
├── tag_matcher.py       # TagMatcher: 三层匹配引擎
└── tests/
    ├── test_tag_store.py
    ├── test_tag_builder.py
    └── test_tag_matcher.py
```
