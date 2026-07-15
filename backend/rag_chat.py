import math
import os
import pickle

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from FlagEmbedding import FlagReranker

from retrieval.bm25_search import BM25Retriever
from retrieval.hybrid_search import HybridRetriever


load_dotenv()


# ======================
# 1. 初始化Embedding
# ======================

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)

# ======================
# Reranker模型
# ======================

reranker = FlagReranker(
    "BAAI/bge-reranker-base",
    use_fp16=True
)

# ======================
# 2. 加载向量库
# ======================

db = FAISS.load_local(
    "./vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)


# ======================
# 3. 创建混合检索器
# ======================

with open(
    "./vectorstore/docs.pkl",
    "rb"
) as f:
    docs = pickle.load(f)

bm25 = BM25Retriever(
    docs
)

vector_retriever = db.as_retriever(
    search_kwargs={
        "k":20
    }
)

retriever = HybridRetriever(
    vector_retriever,
    bm25
)


# ======================
# 4. 加载大模型
# ======================

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.2
)


# ======================
# 检索与拒答配置
# ======================

# Reranker 最终保留的文档数量
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

# 相关度低于该值时直接拒答
# 相关度由原始 Reranker 分数经过 sigmoid 转换得到，范围为 0～1
RERANK_REJECT_THRESHOLD = float(
    os.getenv("RERANK_REJECT_THRESHOLD", "0.50")
)

# 是否在控制台打印检索调试信息
RETRIEVAL_DEBUG = os.getenv(
    "RETRIEVAL_DEBUG",
    "true"
).lower() == "true"


def normalize_rerank_score(raw_score):
    """
    将 Reranker 原始分数通过 sigmoid 转换为 0～1 的相关度。

    FlagReranker 的原始输出通常是未归一化分数，
    可能为负数或大于 1，因此不能直接当作概率使用。
    """

    raw_score = float(raw_score)

    # 防止 exp 溢出
    if raw_score >= 0:
        return 1.0 / (1.0 + math.exp(-raw_score))

    exp_value = math.exp(raw_score)
    return exp_value / (1.0 + exp_value)



def rag_answer(question):
    question = question.strip()

    if not question:
        return {
            "answer": "请输入有效问题。",
            "sources": []
        }

    # ======================
    # 1. Hybrid Search 候选召回
    # ======================

    candidate_docs = retriever.search(
        question,
        k=10
    )

    if not candidate_docs:
        return {
            "answer": "知识库中未找到与该问题相关的医学资料。",
            "sources": []
        }

    # ======================
    # 2. 构造 Reranker 输入
    # ======================

    pairs = [
        [question, doc.page_content]
        for doc in candidate_docs
    ]

    if not pairs:
        return {
            "answer": "知识库中未找到与该问题相关的医学资料。",
            "sources": []
        }

    # ======================
    # 3. Reranker 打分
    # ======================

    raw_scores = reranker.compute_score(pairs)

    # 当只有一个候选时，部分版本可能返回单个 float
    if isinstance(raw_scores, (int, float)):
        raw_scores = [raw_scores]

    ranked_items = []

    for doc, raw_score in zip(candidate_docs, raw_scores):
        relevance = normalize_rerank_score(raw_score)

        ranked_items.append({
            "doc": doc,
            "raw_score": float(raw_score),
            "relevance": relevance
        })

    ranked_items.sort(
        key=lambda item: item["relevance"],
        reverse=True
    )

    if not ranked_items:
        return {
            "answer": "知识库中未找到与该问题相关的医学资料。",
            "sources": []
        }

    # ======================
    # 4. 控制台调试输出
    # ======================

    if RETRIEVAL_DEBUG:
        for index, item in enumerate(
            ranked_items[:RERANK_TOP_N],
            start=1
        ):
            doc = item["doc"]
            page = doc.metadata.get("page", -1)

            page_text = (
                str(page + 1)
                if isinstance(page, int) and page >= 0
                else "未知"
            )

            print("\n==========")
            print(f"第 {index} 个结果")
            print(f"页码: {page_text}")
            print(f"Reranker 原始分数: {item['raw_score']:.4f}")
            print(f"归一化相关度: {item['relevance']:.4f}")
            print(doc.page_content[:500])

    # ======================
    # 5. 相关度阈值硬拒答
    # ======================

    highest_relevance = ranked_items[0]["relevance"]

    if highest_relevance < RERANK_REJECT_THRESHOLD:
        return {
            "answer": "知识库中未找到与该问题相关的医学资料。",
            "sources": []
        }

    # ======================
    # 6. 选取达到阈值的 Top-N 文档
    # ======================

    accepted_items = [
        item
        for item in ranked_items
        if item["relevance"] >= RERANK_REJECT_THRESHOLD
    ][:RERANK_TOP_N]

    if not accepted_items:
        return {
            "answer": "知识库中未找到与该问题相关的医学资料。",
            "sources": []
        }

    docs = [
        item["doc"]
        for item in accepted_items
    ]

    # 拼接资料
    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )


    prompt = f"""
你是一名医疗知识助手。

请严格根据下面提供的医学资料回答。

要求：
1. 不要编造资料中没有的信息
2. 如果资料不存在答案，请明确说明
3. 回答简洁准确


医学资料：

{context}


用户问题：

{question}

"""


    response = llm.invoke(prompt)


    # 来源仅使用通过阈值的文档，并保留 Reranker 排序
    sources = []
    seen_sources = set()

    for item in accepted_items:
        doc = item["doc"]
        relevance = item["relevance"]

        page = doc.metadata.get("page", "未知")

        if isinstance(page, int):
            page_text = str(page + 1)
        else:
            page_text = str(page)

        source_name = doc.metadata.get(
            "source",
            "糖尿病指南.pdf"
        )

        source_text = (
            f"{source_name} 第{page_text}页"
            f"（相关度 {relevance:.3f}）"
        )

        if source_text not in seen_sources:
            seen_sources.add(source_text)
            sources.append(source_text)

    return {
        "answer":response.content,
        "sources":sources
    }



# 测试

if __name__=="__main__":

    question="糖尿病神经病变应该如何筛查？"


    result=rag_answer(question)


    print("\n回答:")
    print(result["answer"])


    print("\n来源:")
    for s in result["sources"]:
        print("-",s)
