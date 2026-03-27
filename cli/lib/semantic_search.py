from sentence_transformers import SentenceTransformer
from constants import SCORE_PRECISION
from pathlib import Path
import numpy as np
import json, re

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    with open('data/movies.json', 'rb') as f:
        documents = json.load(f)
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def embed_query(query):
    semantic_search = SemanticSearch()
    return semantic_search.generate_embedding(query)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_chunk(text, max_chunk_size, overlap):
        text_stripped = text.strip()
        if text_stripped == "":
            return []
        words = re.split(r"(?<=[.!?])\s+", text_stripped)
        if len(words) == 1 and not words[0].endswith(('?', '.', '!')):
            words = [text_stripped]
        for i in range(len(words)):
            words[i] = words[i].strip()
        words = [w for w in words if w != ""]
        step = max_chunk_size-overlap
        n = max_chunk_size
        chunks = []
        i = 0
        while i < len(words):
            window = words[i : i + n]
            # Stop if we already have chunks and this window is too small
            if chunks and len(window) <= overlap:
                break
            chunks.append(" ".join(window))
            i += step
        #chunks = [" ".join(words[i: i+n]) for i in range(0, len(words), step)]
        return chunks


class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer(model_name, device='cpu')
        self.embeddings = None #embeddings of the model
        self.documents = None #list of dictionaries. Each dict is a movie
        self.document_map = dict() #maps id to full document

    def generate_embedding(self, text: str):
        if text.isspace():
            raise ValueError("string should not be empty")
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        strList = []
        
        for document in self.documents['movies']:
            self.document_map[document['id']] = document
            strList.append(f"{document["title"]}: {document['description']}")
        
        self.embeddings = self.model.encode(strList, show_progress_bar = True)
        
        with open('cache/movie_embeddings.npy', 'wb') as f:
            np.save(f, self.embeddings)
        
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for document in self.documents['movies']:
            self.document_map[document['id']] = document

        file_path = Path('cache/movie_embeddings.npy')

        if not file_path.exists():
            return self.build_embeddings(documents)
        
        with open('cache/movie_embeddings.npy', 'rb') as f:
            self.embeddings = np.load(f)

        if len(self.embeddings) != len(self.documents['movies']):
            raise Exception("length of embeddings different from length of documents")

        return self.embeddings
    
    def search(self, query, limit):
        
        file_path = Path('cache/movie_embeddings.npy')

        if not file_path.exists():
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        similarity_list = []

        for i in range(len(self.embeddings)):
            similarity_list.append((cosine_similarity(query_embedding, self.embeddings[i]), self.documents['movies'][i]))
        
        similarity_list.sort(key=lambda x: x[0], reverse=True)
        top_results = similarity_list[:limit]
        results = []
        
        for entry in top_results:
            results.append({
                "score": entry[0],
                "title": entry[1]["title"],
                "description": entry[1]["description"]
                })
        
        return results
            
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None       

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        allchunks = []
        metadataDict = []
        chunk_index = 0
        self.document_map = {}
        
        for i in range(len(self.documents)):
            self.document_map[self.documents[i]['id']] = self.documents[i]
            if not self.documents[i]['description'].strip() or not self.documents[i]['description']:
                continue
            chunks = semantic_chunk(self.documents[i]['description'], 4, 1)

            for idx, chunk in enumerate(chunks):
                allchunks.append(chunk)
                metadataDict.append({'movie_idx':i, 'chunk_idx':idx, 'total_chunks':len(chunks)})
                chunk_index += 1
            chunk_index = 0
        
        print(len(allchunks))
        print(allchunks[:3])
        self.chunk_embeddings = self.model.encode(allchunks, show_progress_bar=True)
        self.chunk_metadata = metadataDict

        with open('cache/chunk_embeddings.npy', 'wb') as f:
            np.save(f, self.chunk_embeddings)

        with open('cache/chunk_metadata.json', 'w') as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks":len(allchunks)}, f, indent=2)

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for i in range(len(self.documents)):
            self.document_map[self.documents[i]['id']] = self.documents[i]

        filepath1 = Path('cache/chunk_embeddings.npy')
        filepath2 = Path('cache/chunk_metadata.json')

        if not filepath1.exists() or not filepath2.exists():
            return self.build_chunk_embeddings(documents)
        
        with open('cache/chunk_embeddings.npy', 'rb') as f:
            self.chunk_embeddings = np.load(f)

        with open('cache/chunk_metadata.json', 'r') as f:
            self.chunk_metadata = json.load(f)['chunks']

        return self.chunk_embeddings
        
    def search_chunks(self, query: str, limit: int = 10):
        query_chunks = super().generate_embedding(query)
        chunk_scores = []
        
        for idx, embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_chunks, embedding)
            chunk_scores.append({"chunk_idx": idx, "movie_idx":self.chunk_metadata[idx]["movie_idx"],"score": score})

        movie_to_score = dict()
        
        for chunk_score in chunk_scores:
            if chunk_score["movie_idx"] not in movie_to_score or chunk_score["score"] > movie_to_score[chunk_score['movie_idx']]:
                movie_to_score[chunk_score['movie_idx']] = chunk_score["score"]
            
        sorted_items = sorted(movie_to_score.items(),key=lambda item:item[1],reverse=True)[0:limit]
        list_to_return = []
        
        for movie_idx, score in sorted_items:
            doc = self.documents[movie_idx]
            list_to_return.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"],
                "score": round(score, SCORE_PRECISION),
                "metadata": {},
            })
        
        return list_to_return

