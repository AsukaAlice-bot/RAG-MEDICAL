from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from FlagEmbedding import FlagReranker


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
# 3. 创建检索器
# ======================

retriever = db.as_retriever(
    search_kwargs={
        "k":20
    }
)


# ======================
# 4. 加载大模型
# ======================

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.2
)



def rag_answer(question):

    # 检索
    docs = retriever.invoke(question)

    for i, doc in enumerate(docs):
      print("\n==========")
      print("第", i+1, "个结果")
      print("页码:", doc.metadata.get("page")+1)
      print(doc.page_content[:500])

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


    # 来源信息

    sources=[]

    for doc in docs:

        page = doc.metadata.get(
            "page",
            "未知"
        )

        sources.append(
            f"糖尿病指南.pdf 第{page+1}页"
        )


    return {
        "answer":response.content,
        "sources":list(set(sources))
    }



# 测试

if __name__=="__main__":

    question="根据指南，糖尿病慢性并发症包括哪些疾病？请列出具体名称。"


    result=rag_answer(question)


    print("\n回答:")
    print(result["answer"])


    print("\n来源:")
    for s in result["sources"]:
        print("-",s)