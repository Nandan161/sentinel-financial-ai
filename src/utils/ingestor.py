import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.redactor import FinancialRedactor

class FinancialIngestor:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.redactor = FinancialRedactor()
        
        # This is what makes it a 'RAG' system: breaking text into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

    def ingest_document(self, filename):
        file_path = os.path.join(self.raw_dir, filename)
        reader = PdfReader(file_path)
        
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        # 1. Privacy Layer
        print(f"--- Redacting {filename} ---")
        clean_text = self.redactor.redact_text(full_text)
        
        # 2. RAG Preparation Layer (Chunking)
        print(f"--- Chunking {filename} ---")
        chunks = self.splitter.split_text(clean_text)
            
        return chunks

if __name__ == "__main__":
    ingestor = FinancialIngestor()
    chunks = ingestor.ingest_document("tesla_10k.pdf") 
    print(f"✅ Success! Created {len(chunks)} secure chunks.")
    print(f"Sample Chunk: {chunks[0][:200]}...")