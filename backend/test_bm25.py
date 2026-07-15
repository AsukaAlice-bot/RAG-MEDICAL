import pickle

from retrieval.bm25_search import BM25Retriever



with open(
    "./vectorstore/docs.pkl",
    "rb"
) as f:

    docs = pickle.load(f)



bm25 = BM25Retriever(
    docs
)



results = bm25.search(
    "糖尿病慢性并发症",
    5
)


for i,r in enumerate(results):

    print("================")

    print(
        "第",
        i+1,
        "条"
    )

    print(
        r.page_content[:300]
    )