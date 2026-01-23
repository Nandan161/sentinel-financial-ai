import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.redactor import FinancialRedactor

class FinancialIngestor:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.redactor = FinancialRedactor()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def ingest_document(self, file_name):
        file_path = os.path.join("data/raw", file_name)
        
        # 1. Load PDF using PyPDFLoader to keep Metadata (Page numbers)
        loader = PyPDFLoader(file_path)
        pages = loader.load() # This returns a list of Document objects
        
        # 2. Privacy Layer: Redact each page individually to preserve metadata
        print(f"--- Redacting & Cleaning {file_name} ---")
        for page in pages:
            # We redact the content but keep the 'metadata' dict attached to the object
            page.page_content = self.redactor.redact_text(page.page_content)
        
        # 3. RAG Preparation: Split into chunks
        # split_documents ensures every chunk inherits the 'page' and 'source' from the page it came from
        print(f"--- Creating Secure Chunks for {file_name} ---")
        chunks = self.text_splitter.split_documents(pages)
        
        return chunks

if __name__ == "__main__":
    # Test block
    ingestor = FinancialIngestor()
    # Ensure you have a file in data/raw to test
    try:
        chunks = ingestor.ingest_document("tesla_10k.pdf") 
        print(f"✅ Success! Created {len(chunks)} secure chunks.")
        if chunks:
            print(f"Metadata Check (Page): {chunks[0].metadata.get('page')}")
            print(f"Sample Content: {chunks[0].page_content[:150]}...")
    except Exception as e:
        print(f"Could not find test file: {e}")