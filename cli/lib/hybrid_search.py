import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

def normalize(lst_to_normalize: list[float]) -> list[float]:
    if lst_to_normalize == []:
        return []

    M = max(lst_to_normalize)
    m = min(lst_to_normalize)

    if m == M: 
        return [1.0 for _ in lst_to_normalize]
    return [(x - m) / (M - m) for x in lst_to_normalize]

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")