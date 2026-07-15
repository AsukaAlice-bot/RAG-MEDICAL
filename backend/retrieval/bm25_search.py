import jieba

from rank_bm25 import BM25Okapi



class BM25Retriever:


    def __init__(self, docs):

        self.docs = docs


        # 中文分词
        corpus = []

        for doc in docs:

            words = list(
                jieba.cut(
                    doc.page_content
                )
            )

            corpus.append(words)


        # 创建BM25模型
        self.bm25 = BM25Okapi(
            corpus
        )



    def search(
        self,
        query,
        k=5
    ):


        # 查询分词

        query_words = list(
            jieba.cut(query)
        )


        # BM25评分

        scores = self.bm25.get_scores(
            query_words
        )


        # 取最高分

        indexs = sorted(
            range(len(scores)),
            key=lambda i:scores[i],
            reverse=True
        )[:k]


        return [
            self.docs[i]
            for i in indexs
        ]