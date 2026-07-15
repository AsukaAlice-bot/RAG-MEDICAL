from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

import pickle

# PDF路径
pdf_path = "./data/diabetes.pdf"


# 1.读取PDF

loader = PyPDFLoader(pdf_path)

documents = loader.load()

# 删除前10页目录和说明
documents = documents[10:]

print("PDF页数:", len(documents))


# 2.文本切分

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


chunks = splitter.split_documents(documents)


print("文本块数量:", len(chunks))


# 3.加载中文Embedding模型

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)


# 4.创建FAISS数据库

db = FAISS.from_documents(
    chunks,
    embeddings
)

# 保存文本chunks

with open(
    "./vectorstore/docs.pkl",
    "wb"
) as f:

    pickle.dump(
        chunks,
        f
    )

# 5.保存

db.save_local("./vectorstore")


print("======向量库创建完成======")