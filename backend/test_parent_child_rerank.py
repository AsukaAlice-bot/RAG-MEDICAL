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
# 加载环境变量
# ======================

load_dotenv(BASE_DIR / ".env")


# ======================
# 测试配置
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

DEBUG_TOP_N = int(
    os.getenv(
        "PARENT_CHILD_DEBUG_TOP_N",
        "10",
    )
)


def normalize_rerank_score(raw_score):
    """
    将 Reranker 原始分数通过 sigmoid 转换到 0～1。
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
    将 PyPDFLoader 的 0 基页码转换为用户看到的 1 基页码。
    """
    page = metadata.get(
        "page",
        "未知",
    )

    if isinstance(page, int):
        return str(page + 1)

    return str(page)


def load_embeddings():
    """
    使用与现有项目一致的 BGE Embedding。
    """
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def load_reranker():
    """
    加载当前项目已经使用的 BGE Reranker。
    CPU 环境下关闭 fp16。
    """
    return FlagReranker(
        "BAAI/bge-reranker-base",
        use_fp16=False,
    )


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


def rerank_child_documents(
    query,
    child_docs,
    reranker,
):
    """
    使用原始问题对 Child 候选进行精排。
    """
    if not child_docs:
        return []

    pairs = [
        [
            query,
            child_doc.page_content,
        ]
        for child_doc in child_docs
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

    for original_rank, (
        child_doc,
        raw_score,
    ) in enumerate(
        zip(
            child_docs,
            raw_scores,
        ),
        start=1,
    ):
        ranked_items.append({
            "child_doc": child_doc,
            "faiss_rank": original_rank,
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


def select_parent_documents(
    ranked_items,
    parent_store,
):
    """
    过滤低相关 Child，并按 Reranker 顺序返回不同 Parent。
    同一 Parent 只保留最高排名的 Child。
    """
    results = []
    seen_parent_ids = set()

    for rerank_index, item in enumerate(
        ranked_items,
        start=1,
    ):
        if (
            item["relevance"]
            < RERANK_THRESHOLD
        ):
            continue

        child_doc = item["child_doc"]

        parent_id = (
            child_doc.metadata.get(
                "parent_id"
            )
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

        seen_parent_ids.add(
            parent_id
        )

        results.append({
            **item,
            "rerank_rank": rerank_index,
            "parent_id": parent_id,
            "parent_doc": parent_doc,
        })

        if (
            len(results)
            >= PARENT_TOP_N
        ):
            break

    return results


def print_rerank_debug(
    ranked_items,
):
    """
    打印 Reranker 排名前若干条 Child，
    便于观察阈值是否合理。
    """
    print("\n" + "=" * 60)
    print("Child Reranker 排名")
    print(
        f"拒答/过滤阈值："
        f"{RERANK_THRESHOLD:.3f}"
    )

    for index, item in enumerate(
        ranked_items[:DEBUG_TOP_N],
        start=1,
    ):
        child_doc = item["child_doc"]

        print("\n" + "-" * 60)
        print(f"Reranker 排名：{index}")
        print(
            f"FAISS 原始排名："
            f"{item['faiss_rank']}"
        )
        print(
            f"Child 页码："
            f"{format_page(child_doc.metadata)}"
        )
        print(
            f"parent_id："
            f"{child_doc.metadata.get('parent_id')}"
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


def print_parent_results(
    query,
    results,
):
    print("\n" + "=" * 60)
    print(f"查询：{query}")
    print(
        f"返回 Parent 数量："
        f"{len(results)}"
    )

    if not results:
        print(
            "没有 Child 通过相关度阈值，"
            "未返回 Parent。"
        )
        return

    for index, result in enumerate(
        results,
        start=1,
    ):
        child_doc = result["child_doc"]
        parent_doc = result["parent_doc"]

        print("\n" + "=" * 60)
        print(f"Parent 结果 {index}")
        print(
            f"parent_id："
            f"{result['parent_id']}"
        )
        print(
            f"FAISS 原始排名："
            f"{result['faiss_rank']}"
        )
        print(
            f"Reranker 排名："
            f"{result['rerank_rank']}"
        )
        print(
            f"原始分数："
            f"{result['raw_score']:.4f}"
        )
        print(
            f"归一化相关度："
            f"{result['relevance']:.4f}"
        )
        print(
            f"Child 页码："
            f"{format_page(child_doc.metadata)}"
        )
        print(
            f"Parent 页码："
            f"{format_page(parent_doc.metadata)}"
        )

        print("\n命中的 Child：")
        print(
            child_doc.page_content[:500]
        )

        print("\n返回的 Parent：")
        print(
            parent_doc.page_content[:1500]
        )


def main():
    query = (
        "糖尿病神经病变应该如何筛查？"
    )

    embeddings = load_embeddings()

    vectorstore = load_child_vectorstore(
        embeddings
    )

    parent_store = load_parent_store()

    reranker = load_reranker()

    child_docs = (
        vectorstore.similarity_search(
            query,
            k=CHILD_RETRIEVE_K,
        )
    )

    ranked_items = (
        rerank_child_documents(
            query=query,
            child_docs=child_docs,
            reranker=reranker,
        )
    )

    print_rerank_debug(
        ranked_items
    )

    results = select_parent_documents(
        ranked_items=ranked_items,
        parent_store=parent_store,
    )

    print_parent_results(
        query=query,
        results=results,
    )


if __name__ == "__main__":
    main()
