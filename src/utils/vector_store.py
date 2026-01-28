import os
import logging
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import chromadb
from chromadb import PersistentClient
from config import config

logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Custom exception for vector store operations"""
    pass

class FinancialVectorStore:
    def __init__(self, persist_dir: Optional[str] = None):
        """
        Initialize the vector store with enhanced error handling
        
        Args:
            persist_dir: Directory for ChromaDB persistence (uses config default if None)
        """
        self.persist_dir = persist_dir or config.CHROMA_DIR
        
        try:
            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
            logger.info(f"Initialized embeddings with model: {config.EMBEDDING_MODEL}")
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            logger.info(f"Connected to ChromaDB at: {self.persist_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
            raise VectorStoreError(f"Initialization failed: {e}")
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists and has documents
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            True if collection exists and has documents, False otherwise
        """
        # Sanitize collection name
        safe_name = self._sanitize_collection_name(collection_name)
        if not safe_name:
            logger.warning(f"Invalid collection name: {collection_name}")
            return False
        
        try:
            collection_names = [c.name for c in self.client.list_collections()]
            
            if safe_name not in collection_names:
                logger.debug(f"Collection '{safe_name}' does not exist")
                return False
            
            # Check if collection has documents
            col = self.client.get_collection(name=safe_name)
            count = col.count()
            
            logger.debug(f"Collection '{safe_name}' has {count} documents")
            return count > 0
            
        except Exception as e:
            logger.error(f"Error checking collection '{safe_name}': {e}")
            return False
    
    def get_collection_stats(self, collection_name: str) -> dict:
        """
        Get statistics about a collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            if not self.collection_exists(collection_name):
                return {
                    'exists': False,
                    'count': 0,
                    'name': collection_name
                }
            
            col = self.client.get_collection(name=collection_name)
            return {
                'exists': True,
                'count': col.count(),
                'name': collection_name,
                'metadata': col.metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting stats for '{collection_name}': {e}")
            return {
                'exists': False,
                'count': 0,
                'name': collection_name,
                'error': str(e)
            }
    
    def create_store(self, chunks: List[Document], collection_name: str) -> Chroma:
        """
        Create a new vector store from document chunks
        
        Args:
            chunks: List of document chunks to index
            collection_name: Name for the collection
            
        Returns:
            Chroma vector store instance
            
        Raises:
            VectorStoreError: If creation fails
        """
        if not chunks:
            raise VectorStoreError("Cannot create store from empty chunks list")
        
        try:
            logger.info(f"Creating vector store '{collection_name}' with {len(chunks)} chunks")
            
            # Add metadata to track creation
            for chunk in chunks:
                if 'indexed_at' not in chunk.metadata:
                    from datetime import datetime
                    chunk.metadata['indexed_at'] = datetime.now().isoformat()
            
            # Create the vector store
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name=collection_name,
                collection_metadata={
                    "created_at": chunk.metadata.get('indexed_at'),
                    "num_documents": len(chunks)
                }
            )
            
            logger.info(f"Successfully created vector store '{collection_name}'")
            return vector_db
            
        except Exception as e:
            logger.error(f"Failed to create vector store '{collection_name}': {e}", exc_info=True)
            raise VectorStoreError(f"Store creation failed: {e}")
    
    def get_retriever(self, collection_name: str, k: Optional[int] = None):
        """
        Get a retriever for a specific collection
        
        Args:
            collection_name: Name of the collection
            k: Number of documents to retrieve (uses config default if None)
            
        Returns:
            Retriever instance
            
        Raises:
            VectorStoreError: If collection doesn't exist or retrieval fails
        """
        k = k or config.RETRIEVAL_K
        
        if not self.collection_exists(collection_name):
            raise VectorStoreError(
                f"Collection '{collection_name}' does not exist. "
                f"Please index the document first."
            )
        
        try:
            logger.debug(f"Creating retriever for '{collection_name}' with k={k}")
            
            vector_db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            
            return vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
        except Exception as e:
            logger.error(f"Failed to create retriever for '{collection_name}': {e}", exc_info=True)
            raise VectorStoreError(f"Retriever creation failed: {e}")
    
    def get_multi_collection_retriever(self, collection_names: List[str], k: Optional[int] = None):
        """
        Create a retriever that searches across multiple collections
        
        Args:
            collection_names: List of collection names to search
            k: Number of documents to retrieve per collection
            
        Returns:
            Custom multi-collection retriever
            
        Raises:
            VectorStoreError: If any collection doesn't exist
        """
        k = k or config.RETRIEVAL_K
        
        # Validate all collections exist
        missing = [name for name in collection_names if not self.collection_exists(name)]
        if missing:
            raise VectorStoreError(
                f"Collections not found: {', '.join(missing)}. "
                f"Please index these documents first."
            )
        
        try:
            logger.info(f"Creating multi-collection retriever for: {collection_names}")
            
            # Create retrievers for each collection
            retrievers = []
            for name in collection_names:
                retriever = self.get_retriever(name, k=k)
                retrievers.append((name, retriever))
            
            return MultiCollectionRetriever(retrievers, k=k)
            
        except Exception as e:
            logger.error(f"Failed to create multi-collection retriever: {e}", exc_info=True)
            raise VectorStoreError(f"Multi-retriever creation failed: {e}")
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection
        
        Args:
            collection_name: Name of collection to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if not self.collection_exists(collection_name):
                logger.warning(f"Collection '{collection_name}' does not exist")
                return False
            
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False
    
    def list_collections(self) -> List[dict]:
        """
        List all collections with their statistics
        
        Returns:
            List of dictionaries with collection info
        """
        try:
            collections = []
            for col in self.client.list_collections():
                stats = self.get_collection_stats(col.name)
                collections.append(stats)
            
            return collections
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def get_total_documents(self) -> int:
        """
        Get total number of documents across all collections
        
        Returns:
            Total document count
        """
        try:
            total = sum(col['count'] for col in self.list_collections())
            return total
        except Exception as e:
            logger.error(f"Failed to get total documents: {e}")
            return 0
    
    def _sanitize_collection_name(self, name: str) -> str:
        """
        Sanitize collection name to prevent injection attacks
        
        Args:
            name: Collection name to sanitize
            
        Returns:
            Sanitized collection name or empty string if invalid
        """
        if not name:
            return ""
        
        # Only allow alphanumeric characters and underscores
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '', name)
        
        # Must not be empty and not start with number
        if not safe_name or safe_name[0].isdigit():
            logger.warning(f"Invalid collection name: {name}")
            return ""
        
        return safe_name


class MultiCollectionRetriever:
    """
    Custom retriever that searches across multiple collections
    and merges results intelligently
    """
    
    def __init__(self, retrievers: List[tuple], k: int = 10):
        """
        Initialize multi-collection retriever
        
        Args:
            retrievers: List of (collection_name, retriever) tuples
            k: Number of results to return per collection
        """
        self.retrievers = retrievers
        self.k = k
        logger.info(f"Initialized multi-collection retriever with {len(retrievers)} collections")
    
    def invoke(self, query: str) -> List[Document]:
        """
        Retrieve documents from all collections
        
        Args:
            query: Search query
            
        Returns:
            Merged list of documents from all collections
        """
        all_docs = []
        
        for collection_name, retriever in self.retrievers:
            try:
                logger.debug(f"Searching collection '{collection_name}' for: {query[:50]}...")
                docs = retriever.invoke(query)
                
                # Tag each document with its collection name
                for doc in docs:
                    doc.metadata['collection_name'] = collection_name
                
                all_docs.extend(docs)
                logger.debug(f"Retrieved {len(docs)} docs from '{collection_name}'")
                
            except Exception as e:
                logger.error(f"Error retrieving from '{collection_name}': {e}")
                # Continue with other collections
                continue
        
        # Deduplicate based on content similarity
        deduplicated = self._deduplicate_docs(all_docs)
        
        # Sort by relevance (if available) and limit total results
        # For now, just limit to k * num_collections
        max_results = self.k * len(self.retrievers)
        result = deduplicated[:max_results]
        
        logger.info(f"Multi-collection search returned {len(result)} total documents")
        return result
    
    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """
        Remove duplicate documents based on content
        
        Args:
            docs: List of documents to deduplicate
            
        Returns:
            Deduplicated list
        """
        seen_content = set()
        unique_docs = []
        
        for doc in docs:
            # Use first 100 chars as fingerprint
            fingerprint = doc.page_content[:100]
            
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                unique_docs.append(doc)
        
        if len(unique_docs) < len(docs):
            logger.debug(f"Removed {len(docs) - len(unique_docs)} duplicate documents")
        
        return unique_docs


# ===== Testing & Examples =====

if __name__ == "__main__":
    from src.utils.ingestor import FinancialIngestor
    
    # Test vector store initialization
    print("Testing Vector Store...")
    
    try:
        store = FinancialVectorStore()
        print("✅ Vector store initialized")
        
        # List collections
        collections = store.list_collections()
        print(f"\n📚 Found {len(collections)} collection(s):")
        for col in collections:
            print(f"  - {col['name']}: {col['count']} documents")
        
        # Test with sample document
        print("\n--- Testing Document Indexing ---")
        ingestor = FinancialIngestor()
        
        # This assumes you have a test file
        try:
            chunks = ingestor.ingest_document("tesla_10k.pdf")
            
            collection_name = "tesla10k_test"
            if not store.collection_exists(collection_name):
                store.create_store(chunks, collection_name=collection_name)
                print(f"✅ Created collection '{collection_name}'")
            else:
                print(f"ℹ️  Collection '{collection_name}' already exists")
            
            # Test retrieval
            print("\n--- Testing Retrieval ---")
            retriever = store.get_retriever(collection_name, k=3)
            results = retriever.invoke("What was the revenue?")
            print(f"✅ Retrieved {len(results)} documents")
            
            if results:
                print(f"\nSample result:")
                print(f"  Content: {results[0].page_content[:150]}...")
                print(f"  Metadata: {results[0].metadata}")
                
        except FileNotFoundError:
            print("⚠️  No test file found. Skipping ingestion test.")
        
        # Show total stats
        total = store.get_total_documents()
        print(f"\n📊 Total documents across all collections: {total}")
        
    except VectorStoreError as e:
        print(f"❌ Vector store error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()