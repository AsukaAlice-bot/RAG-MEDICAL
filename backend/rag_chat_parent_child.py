import math
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from FlagEmbedding import FlagReranker
from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from query_rewrite import rewrite_query


# ======================
# 路径配置
# ======================

BASE_DIR = Path(__file__).resolve().parent

VECTORSTORE_DIR = (
    BASE_DIR
    / "vectorstore_parent_child"
)

CHILD_INDEX_DIR = (
    VECTORSTORE_DIR
    / "child_index"
)

PARENT_DOCS_PATH = (
    VECTORSTORE_DIR
    / "parent_docs.pkl"
)


# ======================
# 环境变量
# ======================

load_dotenv(BASE_DIR / ".env")


# ======================
# V2 检索配置
# ======================

CHILD_RETRIEVE_K = int(
    os.getenv(
        "PARENT_CHILD_RETRIEVE_K",
        "20",
    )
)

PARENT_TOP_N = int(
    os.getenv(
        "PARENT_CHILD_TOP_N",
        "3",
    )
)

RERANK_THRESHOLD = float(
    os.getenv(
        "PARENT_CHILD_RERANK_THRESHOLD",
        "0.75",
    )
)

RETRIEVAL_DEBUG = os.getenv(
    "RETRIEVAL_DEBUG",
    "true",
).lower() == "true"

COMPUTE_DEVICE = os.getenv(
    "COMPUTE_DEVICE",
    "cpu",
).strip()

RERANK_DEVICE = os.getenv(
    "RERANK_DEVICE",
    (
        "cuda:0"
        if COMPUTE_DEVICE.startswith("cuda")
        else COMPUTE_DEVICE
    ),
).strip()

RERANK_USE_FP16 = os.getenv(
    "RERANK_USE_FP16",
    (
        "true"
        if COMPUTE_DEVICE.startswith("cuda")
        else "false"
    ),
).lower() == "true"


def normalize_rerank_score(raw_score):
    """
    使用 sigmoid 将 Reranker 原始分数转换为 0～1 相关度。
    """
    raw_score = float(raw_score)

    if raw_score >= 0:
        return 1.0 / (
            1.0 + math.exp(-raw_score)
        )

    exp_value = math.exp(raw_score)

    return exp_value / (
        1.0 + exp_value
    )


def format_page(metadata):
    """
    PyPDFLoader 页码从 0 开始，展示时转换为从 1 开始。
    """
    page = metadata.get(
        "page",
        "未知",
    )

    if isinstance(page, int):
        return str(page + 1)

    return str(page)


def format_source_name(metadata):
    """
    只展示文件名，不展示 ./data/ 等本地路径。
    """
    source = metadata.get(
        "source",
        "糖尿病指南.pdf",
    )

    return Path(str(source)).name


def create_embeddings():
    """
    创建与当前项目一致的 BGE Embedding。
    """
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={
            "device": COMPUTE_DEVICE,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def create_reranker():
    """
    创建当前项目使用的 BGE Reranker。

    CPU 环境默认关闭 FP16；
    DCU/GPU 环境默认使用 cuda:0，并开启 FP16。
    """
    return FlagReranker(
        "BAAI/bge-reranker-base",
        use_fp16=RERANK_USE_FP16,
        devices=RERANK_DEVICE,
    )


def create_llm():
    """
    创建兼容 OpenAI 接口的大模型客户端。

    默认使用 qwen-plus；配置 LLM_* 环境变量后，
    可切换到本地 vLLM 模型。
    """
    model = os.getenv(
        "LLM_MODEL",
        "qwen-plus",
    ).strip()

    base_url = os.getenv(
        "LLM_BASE_URL",
        os.getenv("OPENAI_API_BASE", ""),
    ).strip()

    api_key = os.getenv(
        "LLM_API_KEY",
        os.getenv("OPENAI_API_KEY", ""),
    ).strip()

    kwargs = {
        "model": model,
        "temperature": 0.2,
    }

    if base_url:
        kwargs["base_url"] = base_url

    if api_key:
        kwargs["api_key"] = api_key

    return ChatOpenAI(**kwargs)


def load_parent_store():
    if not PARENT_DOCS_PATH.exists():
        raise FileNotFoundError(
            "未找到 parent_docs.pkl，"
            "请先运行 "
            "build_parent_child_vectorstore.py"
        )

    with open(
        PARENT_DOCS_PATH,
        "rb",
    ) as file:
        parent_store = pickle.load(file)

    if not isinstance(
        parent_store,
        dict,
    ):
        raise TypeError(
            "parent_docs.pkl 格式错误，"
            "预期为 dict[str, Document]。"
        )

    return parent_store


def load_child_vectorstore(
    embeddings,
):
    if not CHILD_INDEX_DIR.exists():
        raise FileNotFoundError(
            "未找到 Child FAISS 索引，"
            "请先运行 "
            "build_parent_child_vectorstore.py"
        )

    return FAISS.load_local(
        str(CHILD_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ======================
# 初始化
# ======================

embeddings = create_embeddings()

child_vectorstore = load_child_vectorstore(
    embeddings
)

parent_store = load_parent_store()

reranker = create_reranker()

llm = create_llm()


def retrieve_child_candidates(
    search_queries,
):
    """
    对原始问题和改写问题分别执行 Child FAISS 检索，
    再合并并去重候选 Child。
    """
    candidate_docs = []
    seen_candidates = set()

    for search_query in search_queries:
        current_docs = (
            child_vectorstore.similarity_search(
                search_query,
                k=CHILD_RETRIEVE_K,
            )
        )

        for child_doc in current_docs:
            key = (
                child_doc.metadata.get(
                    "child_id",
                    "",
                ),
                child_doc.metadata.get(
                    "parent_id",
                    "",
                ),
                child_doc.metadata.get(
                    "source",
                    "",
                ),
                child_doc.metadata.get(
                    "page",
                    -1,
                ),
                child_doc.page_content.strip(),
            )

            if key in seen_candidates:
                continue

            seen_candidates.add(key)
            candidate_docs.append(child_doc)

    return candidate_docs


def rerank_child_candidates(
    question,
    candidate_docs,
):
    """
    始终使用用户原始问题对 Child 候选进行 Reranker 打分。
    """
    if not candidate_docs:
        return []

    pairs = [
        [
            question,
            child_doc.page_content,
        ]
        for child_doc in candidate_docs
    ]

    raw_scores = reranker.compute_score(
        pairs
    )

    if isinstance(
        raw_scores,
        (int, float),
    ):
        raw_scores = [raw_scores]

    ranked_items = []

    for child_doc, raw_score in zip(
        candidate_docs,
        raw_scores,
    ):
        ranked_items.append({
            "child_doc": child_doc,
            "raw_score": float(
                raw_score
            ),
            "relevance": (
                normalize_rerank_score(
                    raw_score
                )
            ),
        })

    ranked_items.sort(
        key=lambda item: (
            item["relevance"]
        ),
        reverse=True,
    )

    return ranked_items


def select_parent_contexts(
    ranked_items,
):
    """
    过滤低相关 Child，并按 Child 排名返回不同 Parent。
    同一 Parent 只保留最高相关 Child。
    """
    accepted_parents = []
    seen_parent_ids = set()

    for rerank_rank, item in enumerate(
        ranked_items,
        start=1,
    ):
        if (
            item["relevance"]
            < RERANK_THRESHOLD
        ):
            continue

        child_doc = item["child_doc"]

        parent_id = child_doc.metadata.get(
            "parent_id"
        )

        if not parent_id:
            continue

        if parent_id in seen_parent_ids:
            continue

        parent_doc = parent_store.get(
            parent_id
        )

        if parent_doc is None:
            continue

        seen_parent_ids.add(parent_id)

        accepted_parents.append({
            **item,
            "rerank_rank": rerank_rank,
            "parent_id": parent_id,
            "parent_doc": parent_doc,
        })

        if (
            len(accepted_parents)
            >= PARENT_TOP_N
        ):
            break

    return accepted_parents


def print_retrieval_debug(
    question,
    rewritten_query,
    search_queries,
    ranked_items,
    accepted_parents,
):
    if not RETRIEVAL_DEBUG:
        return

    print("\n" + "=" * 60)
    print(f"原始问题：{question}")
    print(f"改写查询：{rewritten_query}")
    print(
        f"实际检索查询数："
        f"{len(search_queries)}"
    )
    print(
        f"Child 候选数量："
        f"{len(ranked_items)}"
    )
    print(
        f"Parent 最终数量："
        f"{len(accepted_parents)}"
    )
    print(
        f"相关度阈值："
        f"{RERANK_THRESHOLD:.3f}"
    )

    for index, item in enumerate(
        ranked_items[:10],
        start=1,
    ):
        child_doc = item["child_doc"]

        print("\n" + "-" * 60)
        print(f"Child Reranker 排名：{index}")
        print(
            f"parent_id："
            f"{child_doc.metadata.get('parent_id')}"
        )
        print(
            f"页码："
            f"{format_page(child_doc.metadata)}"
        )
        print(
            f"原始分数："
            f"{item['raw_score']:.4f}"
        )
        print(
            f"归一化相关度："
            f"{item['relevance']:.4f}"
        )
        print(
            f"是否通过阈值："
            f"{'是' if item['relevance'] >= RERANK_THRESHOLD else '否'}"
        )
        print(child_doc.page_content[:500])


def build_context(
    accepted_parents,
):
    """
    使用完整 Parent 文本构造大模型上下文。
    """
    context_parts = []

    for index, item in enumerate(
        accepted_parents,
        start=1,
    ):
        parent_doc = item["parent_doc"]

        source_name = format_source_name(
            parent_doc.metadata
        )

        page_text = format_page(
            parent_doc.metadata
        )

        context_parts.append(
            (
                f"[资料{index}]\n"
                f"来源：{source_name} "
                f"第{page_text}页\n"
                f"相关度："
                f"{item['relevance']:.3f}\n"
                f"正文：\n"
                f"{parent_doc.page_content}"
            )
        )

    return "\n\n".join(context_parts)


def build_sources(
    accepted_parents,
):
    """
    根据通过阈值的 Parent 生成有序来源列表。
    """
    sources = []
    seen_sources = set()

    for item in accepted_parents:
        parent_doc = item["parent_doc"]

        source_name = format_source_name(
            parent_doc.metadata
        )

        page_text = format_page(
            parent_doc.metadata
        )

        source_text = (
            f"{source_name} "
            f"第{page_text}页"
            f"（相关度 "
            f"{item['relevance']:.3f}）"
        )

        if source_text in seen_sources:
            continue

        seen_sources.add(source_text)
        sources.append(source_text)

    return sources


def rag_answer_parent_child(
    question,
):
    """
    父子块 RAG V2 独立问答入口。
    """
    question = str(
        question or ""
    ).strip()

    if not question:
        return {
            "answer": "请输入有效问题。",
            "sources": [],
        }

    # 1. 查询改写
    rewritten_query = rewrite_query(
        llm,
        question,
    )

    search_queries = [question]

    if (
        rewritten_query
        and rewritten_query.strip()
        and rewritten_query.strip()
        != question
    ):
        search_queries.append(
            rewritten_query.strip()
        )

    # 2. Child 多路召回
    candidate_docs = (
        retrieve_child_candidates(
            search_queries
        )
    )

    if not candidate_docs:
        return {
            "answer": (
                "知识库中未找到与该问题"
                "相关的医学资料。"
            ),
            "sources": [],
        }

    # 3. Child Reranker
    ranked_items = (
        rerank_child_candidates(
            question,
            candidate_docs,
        )
    )

    # 4. 阈值过滤并返回 Parent
    accepted_parents = (
        select_parent_contexts(
            ranked_items
        )
    )

    print_retrieval_debug(
        question=question,
        rewritten_query=rewritten_query,
        search_queries=search_queries,
        ranked_items=ranked_items,
        accepted_parents=accepted_parents,
    )

    if not accepted_parents:
        return {
            "answer": (
                "知识库中未找到与该问题"
                "相关的医学资料。"
            ),
            "sources": [],
        }

    # 5. 使用 Parent 构造上下文
    context = build_context(
        accepted_parents
    )

    prompt = f"""
你是一名严谨的医疗知识库问答助手。

请严格根据下面提供的医学资料回答用户问题。

要求：
1. 只能使用资料中明确出现的信息。
2. 不得编造、推测或扩展资料中没有的内容。
3. 如果资料不足以回答，应明确说明资料不足。
4. 先直接回答用户问题，再补充必要说明。
5. 对同义、重复、上下位重合的项目进行合并，不得重复列举。
6. 只保留与用户问题直接相关的主要结论，不要罗列无关背景信息。
7. 不得把“可能相关、风险增加、评估方法”等内容扩展成独立疾病或结论。
8. 用户询问“包括哪些”时，优先归纳主要类别，每类只说明一次，通常不超过8项。
9. 不要为了增加数量而拆分同一疾病，也不要重复使用不同名称列出同一内容。
10. 不要把检索分数写进回答正文。
11. 本回答仅用于知识查询，不替代医生诊断和治疗。

医学资料：
{context}

用户问题：
{question}
""".strip()

    response = llm.invoke(
        prompt
    )

    sources = build_sources(
        accepted_parents
    )

    return {
        "answer": response.content,
        "sources": sources,
    }


def main():
    question = (
        "糖尿病神经病变应该如何筛查？"
    )

    result = rag_answer_parent_child(
        question
    )

    print("\n" + "=" * 60)
    print("回答：")
    print(result["answer"])

    print("\n来源：")

    if not result["sources"]:
        print("- 无")
    else:
        for source in result["sources"]:
            print(f"- {source}")


if __name__ == "__main__":
    main()
