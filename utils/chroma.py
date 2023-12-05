from chromadb import Documents, EmbeddingFunction, Embeddings

class BGEEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embed_model) -> None:
        self.embed_model = embed_model
        super().__init__()
    
    def __call__(self, texts: Documents) -> Embeddings:
        # embed the documents somehow
        embeddings = self.embed_model.embed_documents(texts)
        return embeddings