from sentence_transformers import SentenceTransformer

class LocalEmbeddingModel:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the SentenceTransformer model."""
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """Generate embeddings for a list of documents."""
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, query):
        """Generate embedding for a single query."""
        return self.model.encode(query, show_progress_bar=False)
