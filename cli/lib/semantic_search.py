from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import json

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

class SemanticSearch:
    def __init__(self):
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
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

        if len(self.embeddings) != len(documents):
            raise Exception("length of embeddings different from length of documents")

        return self.embeddings




