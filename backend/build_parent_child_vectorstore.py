import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ======================
# 路径配置
# ======================

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "diabetes.pdf"

OUTPUT_DIR = BASE_DIR / "vectorstore_parent_child"

CHILD_INDEX_DIR = OUTPUT_DIR / "child_index"

PARENT_DOCS_PATH = OUTPUT_DIR / "parent_docs.pkl"

CHILD_DOCS_PATH = OUTPUT_DIR / "child_docs.pkl"


# ======================
# 切块配置
# ======================

SKIP_PAGES = 10

PARENT_CHUNK_SIZE = 1800
PARENT_CHUNK_OVERLAP = 200

CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100

SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "；",
    "，",
    " ",
    "",
]


def load_source_documents():
    """
    加载 PDF，并跳过前 SKIP_PAGES 页。
    """
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"未找到 PDF 文件：{PDF_PATH}"
        )

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    if len(documents) <= SKIP_PAGES:
        raise ValueError(
            f"PDF 总页数为 {len(documents)}，"
            f"无法跳过前 {SKIP_PAGES} 页。"
        )

    effective_documents = documents[SKIP_PAGES:]

    print(f"PDF 原始页数：{len(documents)}")
    print(f"跳过前 {SKIP_PAGES} 页")
    print(f"参与构建的页数：{len(effective_documents)}")

    return effective_documents


def build_parent_documents(source_documents):
    """
    将原始页面切分成较大的 Parent 文本块。
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=SEPARATORS,
        add_start_index=True,
    )

    parent_documents = parent_splitter.split_documents(
        source_documents
    )

    parent_store = {}

    for parent_index, parent_doc in enumerate(
        parent_documents
    ):
        parent_id = f"parent_{parent_index:06d}"

        parent_doc.metadata["parent_id"] = parent_id
        parent_doc.metadata["chunk_type"] = "parent"
        parent_doc.metadata["parent_index"] = parent_index

        parent_store[parent_id] = parent_doc

    return parent_store


def build_child_documents(parent_store):
    """
    将每个 Parent 文本块继续切分成较小的 Child 文本块。
    每个 Child 通过 parent_id 关联到所属 Parent。
    """
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=SEPARATORS,
        add_start_index=True,
    )

    child_documents = []
    global_child_index = 0

    for parent_id, parent_doc in parent_store.items():
        children = child_splitter.split_documents(
            [parent_doc]
        )

        for local_child_index, child_doc in enumerate(
            children
        ):
            child_id = f"child_{global_child_index:07d}"

            child_doc.metadata["child_id"] = child_id
            child_doc.metadata["parent_id"] = parent_id
            child_doc.metadata["chunk_type"] = "child"
            child_doc.metadata[
                "child_index"
            ] = global_child_index
            child_doc.metadata[
                "local_child_index"
            ] = local_child_index

            child_documents.append(child_doc)
            global_child_index += 1

    return child_documents


def create_embeddings():
    """
    创建与当前项目一致的 BGE Embedding。
    """
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


def save_parent_child_data(
    parent_store,
    child_documents,
):
    """
    保存 Parent 映射和 Child 文档列表。
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(PARENT_DOCS_PATH, "wb") as file:
        pickle.dump(parent_store, file)

    with open(CHILD_DOCS_PATH, "wb") as file:
        pickle.dump(child_documents, file)


def build_child_vectorstore(
    child_documents,
    embeddings,
):
    """
    仅使用 Child 文本块建立 FAISS 索引。
    """
    if not child_documents:
        raise ValueError(
            "Child 文档列表为空，无法建立向量库。"
        )

    vectorstore = FAISS.from_documents(
        child_documents,
        embeddings
    )

    CHILD_INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(CHILD_INDEX_DIR)
    )


def print_summary(
    parent_store,
    child_documents,
):
    """
    打印构建结果摘要，便于用户本地验收。
    """
    print("\n" + "=" * 60)
    print("父子块向量库构建完成")
    print(f"Parent 数量：{len(parent_store)}")
    print(f"Child 数量：{len(child_documents)}")
    print(f"Parent 保存路径：{PARENT_DOCS_PATH}")
    print(f"Child 保存路径：{CHILD_DOCS_PATH}")
    print(f"Child FAISS 路径：{CHILD_INDEX_DIR}")

    if parent_store:
        first_parent_id = next(iter(parent_store))
        first_parent = parent_store[first_parent_id]

        print("\nParent 示例：")
        print(f"parent_id：{first_parent_id}")
        print(
            f"页码："
            f"{first_parent.metadata.get('page', '未知')}"
        )
        print(first_parent.page_content[:500])

    if child_documents:
        first_child = child_documents[0]

        print("\nChild 示例：")
        print(
            f"child_id："
            f"{first_child.metadata.get('child_id')}"
        )
        print(
            f"parent_id："
            f"{first_child.metadata.get('parent_id')}"
        )
        print(first_child.page_content[:300])


def main():
    source_documents = load_source_documents()

    parent_store = build_parent_documents(
        source_documents
    )

    child_documents = build_child_documents(
        parent_store
    )

    save_parent_child_data(
        parent_store,
        child_documents,
    )

    embeddings = create_embeddings()

    build_child_vectorstore(
        child_documents,
        embeddings,
    )

    print_summary(
        parent_store,
        child_documents,
    )


if __name__ == "__main__":
    main()
