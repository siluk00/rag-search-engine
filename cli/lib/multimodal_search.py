from PIL import Image
from sentence_transformers import SentenceTransformer

def verify_image_embedding(path):
    multimodal_search = MultimodalSearch()
    embedding = multimodal_search.embed_image(path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

class MultimodalSearch():
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name, device='cpu')

    def embed_image(self, path):
        img = Image.open("data/paddington.jpeg")
        return self.model.encode(img)