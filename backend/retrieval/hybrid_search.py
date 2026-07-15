class HybridRetriever:


    def __init__(
        self,
        vector_retriever,
        bm25_retriever
    ):

        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever



    def search(self, query, k=10):
        """
        同时执行 FAISS 向量检索和 BM25 关键词检索，
        合并候选文档，并根据来源、页码和正文内容进行可靠去重。

        Args:
            query: 用户问题。
            k: BM25 返回数量。

        Returns:
            去重后的候选 Document 列表。
        """

        # 1. FAISS 向量检索
        vector_docs = self.vector_retriever.invoke(query)

        # 2. BM25 关键词检索
        bm25_docs = self.bm25_retriever.search(query, k)

        # 3. 合并并去重
        unique_docs = []
        seen = set()

        for doc in vector_docs + bm25_docs:
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", -1),
                doc.page_content.strip(),
            )

            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        return unique_docs
