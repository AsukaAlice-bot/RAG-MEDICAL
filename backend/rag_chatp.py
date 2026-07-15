from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# 新增混合检索所需包
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# ======================
# 1. 初始化Embedding
# ======================
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
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
# 3. 创建混合检索器 (核心增强)
# ======================

def get_bm25_retriever():
    """创建 BM25 关键词检索器"""
    loader = PyPDFLoader("./data/diabetes.pdf")
    documents = loader.load()
    # 跳过前10页目录
    documents = documents[10:]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10
    return bm25_retriever


# 向量检索器
vector_retriever = db.as_retriever(search_kwargs={"k": 10})

# BM25 检索器
bm25_retriever = get_bm25_retriever()

# 混合检索器
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]   # 可根据效果调整比例
)


# ======================
# 4. 加载大模型 (千问)
# ======================
llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.2
)


def rag_answer(question):
    """增强版 RAG 问答函数"""
    
    # 使用混合检索
    docs = hybrid_retriever.invoke(question)

    print(f"\n=== 检索到 {len(docs)} 条相关文档 ===")
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "未知") + 1
        print(f"\n第 {i+1} 个结果 | 页码: {page}")
        print(doc.page_content[:600] + "..." if len(doc.page_content) > 600 else doc.page_content)

    # 拼接上下文
    context = "\n\n".join([doc.page_content for doc in docs])

    # 优化后的 Prompt
    prompt = f"""
你是一名严谨的临床医疗知识助手，专门解答糖尿病相关问题。

请严格根据下面提供的《糖尿病防治指南》资料回答用户问题。
要求：
1. 只使用资料中明确存在的内容，不得编造或推测。
2. 如果资料中无法找到答案，请明确回复“根据提供的指南资料，无法找到相关信息。”
3. 回答要简洁、专业、准确。
4. 回答结束后列出主要参考来源页码。

医学资料：
{context}

用户问题：
{question}

请直接给出答案：
"""

    response = llm.invoke(prompt)

    # 来源信息
    sources = []
    for doc in docs:
        page = doc.metadata.get("page", "未知")
        sources.append(f"糖尿病指南.pdf 第{page + 1}页")

    return {
        "answer": response.content,
        "sources": list(set(sources))
    }


# 测试
if __name__ == "__main__":
    question = "根据指南，糖尿病慢性并发症包括哪些疾病？请列出具体名称。"
    
    result = rag_answer(question)
    
    print("\n" + "="*50)
    print("最终回答:")
    print(result["answer"])
    print("\n来源:")
    for s in result["sources"]:
        print("-", s)