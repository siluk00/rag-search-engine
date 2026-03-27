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

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=60):
    return 1 / (k + rank)

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
        literals = self._bm25_search(query, limit*500) #list of dictionaries
        semantics = self.semantic_search.search_chunks(query, limit*500) ##list of dictionaries
        scores = dict()
        literals_normalized = [r['score'] for r in literals]
        semantics_normalized = [r['score'] for r in semantics]
        literals_normalized = normalize(literals_normalized)
        semantics_normalized = normalize(semantics_normalized)
        for i, result in enumerate(literals):
            scores[result['id']] = {
                'id':result['id'],
                'title':result['title'],
                'document':result['document'],
                'bm_25_score':literals_normalized[i],
                'semantic_score':0.0
                }
        for i, result in enumerate(semantics):
            if result['id'] in scores:
                scores[result['id']]['semantic_score']=semantics_normalized[i]
            else:
                scores[result['id']] = {
                    'id': result['id'],
                    'title':result['title'],
                    'document':result['document'],
                    'bm_25_score':0.0,
                    'semantic_score':semantics_normalized[i]
                }
        for key in scores:
            scores[key]['score'] = hybrid_score(scores[key]['bm_25_score'], scores[key]['semantic_score'], alpha)
        list_to_return = sorted(scores.values(), key=lambda x:x['score'], reverse=True)[:limit]
        return list_to_return

    def rrf_search(self, query, k=60, limit=10):
        literals = self._bm25_search(query, 500*limit)
        semantics = self.semantic_search.search_chunks(query, limit*500)
        scores = {}

        for i, result in enumerate(literals):
            scores[result['id']] = {
                'id':result['id'],
                'title':result['title'],
                'document':result['document'],
                'bm_25_score':i,
                'semantic_score':0.0
                }
        for i, result in enumerate(semantics):
            if result['id'] in scores:
                scores[result['id']]['semantic_score']=i
            else:
                scores[result['id']] = {
                    'id': result['id'],
                    'title':result['title'],
                    'document':result['document'],
                    'bm_25_score':0.0,
                    'semantic_score':i
                }
        
        for key in scores:
            scores[key]['rrf_score'] = \
                rrf_score(scores[key]['bm_25_score'], k) + \
                rrf_score(scores[key]['semantic_score'], k)

        list_to_return = sorted(scores.values(), key=lambda x:x['rrf_score'], reverse=True)[:limit] #list of dictionaries

        return list_to_return