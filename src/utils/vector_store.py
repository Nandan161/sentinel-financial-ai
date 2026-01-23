import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class FinancialVectorStore:
    def __init__(self, persist_dir="chroma_db"):
        self.persist_dir = persist_dir
        # Using the model we just pulled in Ollama
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

    def create_store(self, chunks, collection_name="tesla_10k_report"):
        print(f"--- Creating Vector Store for {collection_name} ---")
        
        # This part prevents mixing different PDFs in the same "shelf"
        try:
            from chromadb import PersistentClient
            client = PersistentClient(path=self.persist_dir)
            # Delete old version of this collection if it exists
            client.delete_collection(name=collection_name)
        except:
            pass 

        vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=collection_name
        )
        print(f"✅ Vector Store for {collection_name} Created Successfully!")
        return vector_db

    def get_retriever(self, collection_name="tesla_10k_report"): 
        vector_db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        return vector_db.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    # Test Logic: Connect Ingestor + VectorStore
    from src.utils.ingestor import FinancialIngestor
    
    ingestor = FinancialIngestor()
    chunks = ingestor.ingest_document("tesla_10k.pdf")
    
    store = FinancialVectorStore()
    store.create_store(chunks)