from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, Qdrant
import os

class VectorDB:
    def __init__(self, qclient, model="text-embedding-3-small"):
        self.qclient = qclient
        self.get_embedding_model = OpenAIEmbeddings(model=model)
        self.qdrant_vector = None
    
    def set_embedding_vector(self):
         """use this method to initiatilize the qdrant_vector stor without arguments"""
         self.qdrant_vector = QdrantVectorStore.from_existing_collection(
                collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
                embedding=self.get_embedding_model,
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
            )

    def embed_text(self, docs):
        try:
            self.qdrant_vector = QdrantVectorStore.from_existing_collection(
                collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
                embedding=self.get_embedding_model,
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
            )
        except Exception:
            print("Error connecting to vector store.")
            if self.qdrant_vector is None:
                self.qdrant_vector = QdrantVectorStore.from_documents(
                    documents=docs,
                    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
                    embedding=self.get_embedding_model,
                    url=os.getenv("QDRANT_URL"),
                    api_key=os.getenv("QDRANT_API_KEY"),
                )
        
    def retrieve_similar_docs(self, query, k=5):
        if self.qdrant_vector is None:
            raise ValueError("Vector store is not initialized. Please embed documents first.")
        return self.qdrant_vector.similarity_search(query, k=k)