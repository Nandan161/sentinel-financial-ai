import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class FinancialVectorStore:
    def __init__(self, persist_dir="chroma_db"):
        self.persist_dir = persist_dir
        # Using the model we just pulled in Ollama
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

    def create_store(self, chunks):
        print(f"--- Creating Vector Store in {self.persist_dir} ---")
        
        # This command does 3 things: 
        # 1. Embeds each chunk, 2. Indexes them, 3. Saves to disk
        vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="tesla_10k_report"
        )
        print("✅ Vector Store Created and Persisted Successfully!")
        return vector_db

    def get_retriever(self):
        # Load the existing store
        vector_db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="tesla_10k_report"
        )
        return vector_db.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    # Test Logic: Connect Ingestor + VectorStore
    from src.utils.ingestor import FinancialIngestor
    
    ingestor = FinancialIngestor()
    chunks = ingestor.ingest_document("tesla_10k.pdf")
    
    store = FinancialVectorStore()
    store.create_store(chunks)