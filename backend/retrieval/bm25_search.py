from rank_bm25 import BM25Okapi
import pickle
import jieba


class BM25Retriever:

    def __init__(self, docs):

        self.docs = docs

        # 中文分词
        tokenized_docs = [
            list(jieba.cut(doc.page_content))
            for doc in docs
        ]

        self.bm25 = BM25Okapi(
            tokenized_docs
        )


    def search(self, query, k=5):

        query_words = list(
            jieba.cut(query)
        )

        scores = self.bm25.get_scores(
            query_words
        )


        # 排序
        result_index = sorted(
            range(len(scores)),
            key=lambda i:scores[i],
            reverse=True
        )[:k]


        return [
            self.docs[i]
            for i in result_index
        ]