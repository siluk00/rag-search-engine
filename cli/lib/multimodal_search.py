from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity

def verify_image_embedding(path):
    multimodal_search = MultimodalSearch()
    embedding = multimodal_search.embed_image(path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

class MultimodalSearch():
    def __init__(self, doclist, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name, device='cpu')
        self.doclist = doclist
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in doclist]
        self.text_embeddings = self.model.encode(self.texts)

    def embed_image(self, path):
        img = Image.open("data/paddington.jpeg")
        return self.model.encode(img)

    def search_with_image(self , imgPath):
        img = Image.open("data/paddington.jpeg")
        imgEmbedding = self.model.encode(img)
        similarity = []
        
        for embedding in self.text_embeddings:
            similarity.append(cosine_similarity(embedding, imgEmbedding))
        
        doc_score_pairs = list(zip(self.doclist, similarity))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        top_5_with_scores = []
        for doc, score in doc_score_pairs[:5]:
            top_5_with_scores.append({
                "id": doc.get("id"),
                "title": doc.get("title"),
                "description": doc.get("description"),
                "similarity": float(score)  # Ensure it's a standard float for JSON compatibility
            })

        return top_5_with_scores