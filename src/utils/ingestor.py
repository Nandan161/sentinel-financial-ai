# src/utils/ingestor.py - ENHANCED VERSION

import os
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.redactor import FinancialRedactor
from typing import List, Optional
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass

class FinancialIngestor:
    # Security constants
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'.pdf'}
    
    def __init__(self, chunk_size=1500, chunk_overlap=200):
        """
        Initialize the ingestor with improved chunking parameters
        
        Args:
            chunk_size: Larger size for financial docs with tables
            chunk_overlap: More overlap to preserve context
        """
        self.redactor = FinancialRedactor()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""]  # Better for financial docs
        )
        
        # Statistics tracking
        self.stats = {
            'total_docs': 0,
            'total_pages': 0,
            'total_chunks': 0,
            'total_redactions': 0
        }

    def validate_file(self, file_path: str) -> None:
        """
        Validate file before processing
        
        Raises:
            DocumentProcessingError: If file is invalid
        """
        if not os.path.exists(file_path):
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        # Check extension
        ext = Path(file_path).suffix.lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise DocumentProcessingError(
                f"Invalid file type: {ext}. Allowed: {self.ALLOWED_EXTENSIONS}"
            )
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > self.MAX_FILE_SIZE:
            raise DocumentProcessingError(
                f"File too large: {file_size / 1024 / 1024:.1f}MB. Max: {self.MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        if file_size == 0:
            raise DocumentProcessingError("File is empty")
        
        logger.info(f"File validation passed: {file_path} ({file_size / 1024:.1f}KB)")

    def ingest_document(self, file_name: str) -> Optional[List[Document]]:
        """
        Ingest and process a PDF document with comprehensive error handling
        
        Args:
            file_name: Name of file in data/raw directory
            
        Returns:
            List of processed document chunks, or None if processing fails
            
        Raises:
            DocumentProcessingError: For known processing issues
        """
        file_path = os.path.join("data/raw", file_name)
        
        try:
            # Step 1: Validate
            logger.info(f"Starting ingestion: {file_name}")
            self.validate_file(file_path)
            
            # Step 2: Load PDF
            logger.info("Loading PDF...")
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            if not pages:
                raise DocumentProcessingError("PDF contains no readable pages")
            
            logger.info(f"Loaded {len(pages)} pages")
            
            # Step 3: Redact each page
            logger.info("Applying privacy redaction...")
            redaction_count = 0
            
            for i, page in enumerate(pages):
                try:
                    result = self.redactor.redact_text(
                        page.page_content, 
                        doc_id=f"{file_name}_page_{i}"
                    )
                    page.page_content = result['text']
                    redaction_count += result['redaction_count']
                    
                    if result['entities_found']:
                        logger.info(
                            f"Page {i+1}: Redacted {result['redaction_count']} entities "
                            f"({', '.join(result['entities_found'])})"
                        )
                except Exception as e:
                    logger.error(f"Redaction failed on page {i}: {e}")
                    # Continue with unredacted content rather than failing completely
                    logger.warning(f"Page {i} processed without redaction")
            
            # Step 4: Create chunks
            logger.info("Creating searchable chunks...")
            chunks = self.text_splitter.split_documents(pages)
            
            if not chunks:
                raise DocumentProcessingError("Chunking produced no results")
            
            # Update statistics
            self.stats['total_docs'] += 1
            self.stats['total_pages'] += len(pages)
            self.stats['total_chunks'] += len(chunks)
            self.stats['total_redactions'] += redaction_count
            
            logger.info(
                f"✅ Success: {len(chunks)} chunks created "
                f"({redaction_count} redactions applied)"
            )
            
            return chunks
            
        except DocumentProcessingError as e:
            logger.error(f"Processing failed: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error processing {file_name}: {e}", exc_info=True)
            raise DocumentProcessingError(f"Unexpected error: {str(e)}")
    
    def get_statistics(self) -> dict:
        """Return processing statistics"""
        return self.stats.copy()

if __name__ == "__main__":
    ingestor = FinancialIngestor()
    
    try:
        chunks = ingestor.ingest_document("tesla_10k.pdf")
        print(f"\n{'='*60}")
        print("INGESTION COMPLETE")
        print(f"{'='*60}")
        print(f"Chunks created: {len(chunks)}")
        print(f"\nStatistics:")
        for key, value in ingestor.get_statistics().items():
            print(f"  {key}: {value}")
        
        if chunks:
            print(f"\nSample chunk:")
            print(f"  Page: {chunks[0].metadata.get('page', 'N/A')}")
            print(f"  Source: {Path(chunks[0].metadata.get('source', 'N/A')).name}")
            print(f"  Content preview: {chunks[0].page_content[:200]}...")
            
    except DocumentProcessingError as e:
        print(f"❌ Processing failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")