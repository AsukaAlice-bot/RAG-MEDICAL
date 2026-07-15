import pickle
from pathlib import Path

from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
)
from langchain_community.vectorstores import FAISS


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


def load_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


def load_parent_store():
    if not PARENT_DOCS_PATH.exists():
        raise FileNotFoundError(
            "未找到 parent_docs.pkl，"
            "请先运行 build_parent_child_vectorstore.py"
        )

    with open(PARENT_DOCS_PATH, "rb") as file:
        parent_store = pickle.load(file)

    if not isinstance(parent_store, dict):
        raise TypeError(
            "parent_docs.pkl 格式错误，预期为 dict。"
        )

    return parent_store


def load_child_vectorstore(embeddings):
    if not CHILD_INDEX_DIR.exists():
        raise FileNotFoundError(
            "未找到 Child FAISS 索引，"
            "请先运行 build_parent_child_vectorstore.py"
        )

    return FAISS.load_local(
        str(CHILD_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_parent_documents(
    query,
    vectorstore,
    parent_store,
    child_k=8,
    parent_k=3,
):
    """
    先检索 Child，再根据 parent_id 找回 Parent。
    相同 Parent 只返回一次。
    """
    child_docs = vectorstore.similarity_search(
        query,
        k=child_k,
    )

    results = []
    seen_parent_ids = set()

    for child_rank, child_doc in enumerate(
        child_docs,
        start=1,
    ):
        parent_id = child_doc.metadata.get(
            "parent_id"
        )

        if not parent_id:
            continue

        if parent_id in seen_parent_ids:
            continue

        parent_doc = parent_store.get(parent_id)

        if parent_doc is None:
            continue

        seen_parent_ids.add(parent_id)

        results.append({
            "child_rank": child_rank,
            "child_doc": child_doc,
            "parent_id": parent_id,
            "parent_doc": parent_doc,
        })

        if len(results) >= parent_k:
            break

    return results


def main():
    query = "糖尿病神经病变应该如何筛查？"

    embeddings = load_embeddings()

    parent_store = load_parent_store()

    vectorstore = load_child_vectorstore(
        embeddings
    )

    results = retrieve_parent_documents(
        query=query,
        vectorstore=vectorstore,
        parent_store=parent_store,
        child_k=8,
        parent_k=3,
    )

    print("\n" + "=" * 60)
    print(f"查询：{query}")
    print(f"返回 Parent 数量：{len(results)}")

    if not results:
        print("未找到对应 Parent 文档。")
        return

    for index, result in enumerate(
        results,
        start=1,
    ):
        child_doc = result["child_doc"]
        parent_doc = result["parent_doc"]

        child_page = child_doc.metadata.get(
            "page",
            "未知"
        )

        parent_page = parent_doc.metadata.get(
            "page",
            "未知"
        )

        print("\n" + "=" * 60)
        print(f"结果 {index}")
        print(
            f"Child 原始排名："
            f"{result['child_rank']}"
        )
        print(
            f"parent_id："
            f"{result['parent_id']}"
        )
        print(f"Child 页码：{child_page}")
        print(f"Parent 页码：{parent_page}")

        print("\n命中的 Child：")
        print(child_doc.page_content[:500])

        print("\n返回的 Parent：")
        print(parent_doc.page_content[:1500])


if __name__ == "__main__":
    main()
