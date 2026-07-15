from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_path = "./data/diabetes.pdf"


# 读取PDF
loader = PyPDFLoader(pdf_path)

documents = loader.load()


print("PDF页数:", len(documents))


# 文本切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


chunks = splitter.split_documents(documents)


print("切分数量:", len(chunks))


print("\n第一段内容:")
print(chunks[0].page_content)