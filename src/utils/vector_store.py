import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb
from chromadb import PersistentClient

class FinancialVectorStore:
    def __init__(self, persist_dir="chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # Use PersistentClient to manage collection metadata
        self.client = chromadb.PersistentClient(path=self.persist_dir)

    def collection_exists(self, collection_name):
        """Checks if a collection already exists and has data."""
        try:
            # Check if collection is in the list of existing collections
            collection_names = [c.name for c in self.client.list_collections()]
            if collection_name in collection_names:
                col = self.client.get_collection(name=collection_name)
                return col.count() > 0
            return False
        except Exception:
            return False

    def create_store(self, chunks, collection_name):
        """Creates a new store ONLY if called. Logic for skipping should be in app.py."""
        print(f"--- Indexing Documents for: {collection_name} ---")
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=collection_name
        )
        return vector_db

    def get_retriever(self, collection_name):
        """Dynamically loads the specific collection for a selected document."""
        vector_db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        return vector_db.as_retriever(search_kwargs={"k": 10})

if __name__ == "__main__":
    # Test Logic: Connect Ingestor + VectorStore
    from src.utils.ingestor import FinancialIngestor
    
    ingestor = FinancialIngestor()
    chunks = ingestor.ingest_document("tesla_10k.pdf")
    
    store = FinancialVectorStore()
    store.create_store(chunks)