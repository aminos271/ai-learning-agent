from typing import Any
import re

from core.base_retriever import RetrievedItem



KEYWORD_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "note",
    "notes",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "with",
    "一下",
    "一个",
    "一些",
    "上次",
    "之前",
    "关于",
    "刚刚",
    "刚才",
    "可以",
    "可能",
    "告诉",
    "如何",
    "怎么",
    "我们",
    "我的",
    "找到",
    "有关",
    "最近",
    "有什么",
    "笔记",
    "记录",
    "这次",
    "这个",
    "那个",
}


def rerank_documents(question: str, 
                    items: list[RetrievedItem], 
                    metadata_filter: dict[str, Any] | None = None,
                    top_k: int = 5
                    ) -> list[RetrievedItem]:

    """RAG 检索策略：semantic 为主，keyword 为辅。"""

    if not items:
        return []
    
    ranked_list = []

    
    for item in items:
        semantic_score = float(item.retrieval_meta.get("similarity", 0.0) or 0.0)
        keyword_score = compute_keyword_score(question, item.content)
        metadata_score = 0.0
        
        if metadata_filter:
            matches = 0
            for mk, mv in metadata_filter.items():
                if mk in item.metadata and str(item.metadata.get(mk)) == str(mv):
                    matches += 1
            metadata_score = matches / max(len(metadata_filter), 1)
        
        final_score = semantic_score * 0.65 + keyword_score * 0.25 + metadata_score * 0.1
        ranked_list.append(
            (
            final_score,
            RetrievedItem(
                content=item.content,
                metadata=dict(item.metadata),
                retrieval_meta={
                            **item.retrieval_meta,
                            "similarity": semantic_score,
                            "keyword_score": keyword_score,
                            "metadata_match_score": metadata_score,
                            "rerank_score": final_score,
                        },
            )
            )
        )

    ranked_list.sort(key=lambda item: item[0], reverse=True)
    
    for item in ranked_list[:top_k]:
        
        print(f"排序的分数 {item[0]}\n文档预览 {item[1].content[:40]}")

    return [item[1] for item in ranked_list[:top_k]] 


def extract_keywords(text: str | None) -> list[str]:
    """提取关键词"""
    normalized = normalize_text(text)
    if not normalized:
        return []
    
    keywords = []
    size = 2

    # 处理英文字符
    for token in re.findall(r"[a-z0-9][a-z0-9_+-]{1,}",normalized):
        if token not in KEYWORD_STOPWORDS:
            keywords.append(token)

    # 处理中文字符
    for block in re.findall(r"[\u4e00-\u9fff]+", normalized):
        # 先获取稍微大一点的段落
        if 2 <= len(block) <= 6 and block not in KEYWORD_STOPWORDS:
            keywords.append(block)

            # 分割
            for i in range(len(block) - size + 1):
                token = block[i : i + size]
                if token not in KEYWORD_STOPWORDS:
                    keywords.append(token)

    return list(dict.fromkeys(keywords))


def normalize_text(text: str | None) -> str:
    """标准化文本：去首尾空格 + 全小写 + 多空格压缩成1个空格"""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def compute_keyword_score(query: str, content: str) -> float:
    """计算关键词得分"""
    keywords = extract_keywords(query)
    normalized_content = normalize_text(content)

    if not keywords or not normalized_content:
        return 0.0


    total_hit = 0
    matched_keywords = 0

    # 统计命中次数，最大3次
    for keyword in keywords:
        hit_num = normalized_content.count(keyword)

        if hit_num == 0:
            continue 
        
        matched_keywords += 1
        total_hit += min(hit_num, 3)

    coverage = matched_keywords / len(keywords)
    density = min(total_hit / len(keywords), 1.0)

    return max(0.0, min(coverage * 0.7 + density * 0.3, 1.0))

