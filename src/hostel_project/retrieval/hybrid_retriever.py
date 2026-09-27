from rank_bm25 import BM25Okapi
import re


class HybridRetriever:
    """
    Hybrid retriever combining:
    1. BM25 keyword retrieval
    2. Vector/semantic retrieval
    3. Reciprocal Rank Fusion (RRF)
    """
    def __init__(self, documents, vector_search_function=None):
        """
        Parameters
        documents : list
            List of LangChain Document objects.
        vector_search_function : callable, optional
            Function provided by the vector database layer.
            Expected:
                vector_search_function(query, k)
            and should return:
                [
                    {
                        "text": "...",
                        "metadata": {...},
                        "score": 0.91
                    },
                    ...
                ]
        """
        self.documents = documents
        self.vector_search_function = vector_search_function
        # Prepare documents for BM25
        self.tokenized_documents = [
            self._tokenize(doc.page_content)
            for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_documents)
    # ---------------------------------------------------------
    # TOKENIZATION
    # ---------------------------------------------------------
    @staticmethod
    def _tokenize(text):
        """
        Basic tokenizer for BM25.
        """
        text = text.lower()
        # Keep words and numbers
        tokens = re.findall(r"\b\w+\b", text)
        return tokens
    # ---------------------------------------------------------
    # BM25 SEARCH
    # ---------------------------------------------------------
    def bm25_search(self, query, k=10):
        """
        Perform keyword-based BM25 search.
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        # Get indices sorted by score
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        results = []
        for rank, index in enumerate(ranked_indices[:k], start=1):
            document = self.documents[index]
            results.append({
                "text": document.page_content,
                "metadata": document.metadata,
                "score": float(scores[index]),
                "rank": rank,
                "source": "bm25"
            })
        return results
    # ---------------------------------------------------------
    # VECTOR SEARCH
    # ---------------------------------------------------------
    def vector_search(self, query, k=10):
        """
        Call the vector database/search function.
        """
        if self.vector_search_function is None:
            raise ValueError(
                "vector_search_function has not been provided."
            )
        return self.vector_search_function(query, k)
    # ---------------------------------------------------------
    # RECIPROCAL RANK FUSION
    # ---------------------------------------------------------
    @staticmethod
    def reciprocal_rank_fusion(
        bm25_results,
        vector_results,
        k=60
    ):
        """
        Combine BM25 and vector rankings using RRF.
        RRF formula:
            score = 1 / (k + rank)
        Documents appearing in both rankings
        receive a higher combined score.
        """
        fused_results = {}
        # -------------------------
        # BM25 results
        # -------------------------
        for rank, result in enumerate(bm25_results, start=1):
            key = HybridRetriever._document_key(result)
            if key not in fused_results:
                fused_results[key] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "bm25_rank": None,
                    "vector_rank": None,
                    "rrf_score": 0.0
                }
            fused_results[key]["bm25_rank"] = rank
            fused_results[key]["rrf_score"] += (
                1 / (k + rank)
            )
        # -------------------------
        # Vector results
        # -------------------------
        for rank, result in enumerate(vector_results, start=1):
            key = HybridRetriever._document_key(result)
            if key not in fused_results:
                fused_results[key] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "bm25_rank": None,
                    "vector_rank": None,
                    "rrf_score": 0.0
                }
            fused_results[key]["vector_rank"] = rank
            fused_results[key]["rrf_score"] += (
                1 / (k + rank)
            )
        # Sort according to fused score
        results = sorted(
            fused_results.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        return results
    # ---------------------------------------------------------
    # DOCUMENT ID
    # ---------------------------------------------------------
    @staticmethod
    def _document_key(result):
        """
        Create a stable identifier for a document/chunk.
        Prefer source + page if metadata exists.
        Otherwise fall back to text.
        """
        metadata = result.get("metadata", {})
        source = metadata.get("source")
        page = metadata.get("page")
        if source is not None and page is not None:
            return f"{source}:{page}"
        return result["text"]
    # ---------------------------------------------------------
    # HYBRID SEARCH
    # ---------------------------------------------------------
    def search(self,query,k=5,candidate_k=10):
        """
        Perform complete hybrid retrieval.
        Parameters
        ----------
        query : str
            User query.
        k : int
            Number of final results.
        candidate_k : int
            Number of candidates taken from each retriever.
        """
        # -------------------------
        # BM25
        # -------------------------
        bm25_results = self.bm25_search(
            query,
            k=candidate_k
        )
        # -------------------------
        # Vector search
        # -------------------------
        vector_results = self.vector_search(
            query,
            k=candidate_k
        )
        # -------------------------
        # Fusion
        # -------------------------
        fused_results = self.reciprocal_rank_fusion(
            bm25_results,
            vector_results
        )
        # Return top k
        return fused_results[:k]