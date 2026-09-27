class Reranker:
    """
    Reranks documents returned by the hybrid retriever.
    Input:
        User query + retrieved documents
    Output:
        Same documents ordered by reranker score.
    """
    def __init__(self, rerank_function):
        """
        rerank_function should accept:
            query
            documents
        and return a list of scores.
        """
        self.rerank_function = rerank_function
    def rerank(self, query, documents, top_k=5):
        """
        Rerank retrieved documents.
        Parameters
        ----------
        query : str
            User's question.
        documents : list
            Documents returned by hybrid retrieval.
        top_k : int
            Number of documents to keep after reranking.
        """
        if not documents:
            return []
        # Extract text from retrieved documents
        texts = [
            document["text"]
            for document in documents
        ]
        # Get relevance scores
        scores = self.rerank_function(
            query,
            texts
        )
        # Attach scores
        reranked_documents = []
        for document, score in zip(documents, scores):
            result = document.copy()
            result["rerank_score"] = float(score)
            reranked_documents.append(result)
        # Highest relevance first
        reranked_documents.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        return reranked_documents[:top_k]