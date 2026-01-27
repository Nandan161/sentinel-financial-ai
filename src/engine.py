import logging
from typing import List, Dict, Optional
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from src.utils.vector_store import FinancialVectorStore, VectorStoreError
from config import config
import streamlit as st

logger = logging.getLogger(__name__)

class RAGEngineError(Exception):
    """Custom exception for RAG engine operations"""
    pass

class FinancialRAGEngine:
    def __init__(self):
        """
        Initialize the RAG engine with enhanced capabilities
        """
        try:
            # Initialize vector store
            self.vector_store = FinancialVectorStore()
            logger.info("Vector store initialized")
            
            # Initialize LLM
            self.llm = OllamaLLM(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE
            )
            logger.info(f"LLM initialized: {config.LLM_MODEL}")
            
            # Create prompt template
            self._initialize_prompt()
            
            # Query cache for performance
            self._query_cache = {} if config.CACHE_ENABLED else None
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}", exc_info=True)
            raise RAGEngineError(f"Engine initialization failed: {e}")
    
    def _initialize_prompt(self):
        """
        Initialize the prompt template with enhanced instructions
        """
        self.template = """You are an expert Financial Analyst AI with deep expertise in analyzing financial reports, 10-Ks, 10-Qs, and earnings statements.

Your role is to provide accurate, insightful analysis based ONLY on the provided context documents.

IMPORTANT RULES:
1. DOCUMENT IDENTIFICATION: Each chunk is labeled with ">>> DATA FROM REPORT: [filename] <<<" - use this to identify which company/period the data is from
2. PRIVACY: If you see redacted information (e.g., [PERSON_NAME], [EMAIL], [LOCATION]), keep it redacted in your response
3. MULTI-DOCUMENT COMPARISON: When multiple reports are provided, explicitly compare and contrast the data
4. CITE SOURCES: Always mention which document(s) you're referencing (e.g., "According to the Tesla 10-K...")
5. BE PRECISE: Use exact numbers and metrics from the documents
6. ACKNOWLEDGE LIMITATIONS: If data is missing or unclear, say so
7. NO HALLUCINATION: Never invent data, metrics, or facts not present in the context

CONTEXT DOCUMENTS:
{context}

USER QUESTION:
{question}

RESPONSE GUIDELINES:
- Start with a direct answer
- Cite specific documents when making claims
- If comparing multiple reports, organize by company/period
- Use bullet points for clarity when presenting multiple metrics
- End with relevant caveats or missing information if applicable

YOUR ANALYSIS:"""
        
        self.prompt = ChatPromptTemplate.from_template(self.template)
        logger.debug("Prompt template initialized")
    
    def _get_cache_key(self, query: str, collections: List[str]) -> str:
        """Generate cache key for query"""
        return f"{query}::{','.join(sorted(collections))}"
    
    def _format_context(
        self, 
        docs: List[Document], 
        collection_names: List[str]
    ) -> tuple[str, Dict[str, int]]:
        """
        Format retrieved documents into context with source tracking
        
        Args:
            docs: Retrieved documents
            collection_names: List of collection names being queried
            
        Returns:
            Tuple of (formatted_context, source_stats)
        """
        if not docs:
            return "", {}
        
        # Track document sources
        source_stats = {}
        formatted_chunks = []
        
        for doc in docs:
            # Extract filename from metadata
            source_file = Path(doc.metadata.get("source", "Unknown")).name
            page_num = doc.metadata.get("page", 0) + 1
            
            # Track source statistics
            source_stats[source_file] = source_stats.get(source_file, 0) + 1
            
            # Label each chunk with its source
            chunk_header = f">>> DATA FROM REPORT: {source_file} (Page {page_num}) <<<"
            formatted_chunk = f"{chunk_header}\n{doc.page_content}\n"
            formatted_chunks.append(formatted_chunk)
        
        # Add header with available documents
        available_docs = list(source_stats.keys())
        header = f"Available Reports: {', '.join(available_docs)}\n"
        header += f"Total Context Chunks: {len(docs)}\n"
        header += "\n" + "="*80 + "\n\n"
        
        context_text = header + "\n".join(formatted_chunks)
        
        logger.debug(f"Formatted context from {len(source_stats)} unique sources")
        return context_text, source_stats
    
    def query(
        self, 
        user_question: str, 
        collection_names: List[str],
        k: Optional[int] = None
    ) -> Dict:
        """
        Query the RAG system with enhanced multi-document support
        
        Args:
            user_question: User's question
            collection_names: List of collection names to query
            k: Number of chunks to retrieve (uses config default if None)
            
        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        # Input validation
        if not user_question or not user_question.strip():
            return {
                "answer": "Please provide a question to analyze the documents.",
                "sources": []
            }
        
        if not collection_names:
            return {
                "answer": "No documents selected. Please activate at least one document from the sidebar.",
                "sources": []
            }
        
        # Ensure list format
        if isinstance(collection_names, str):
            collection_names = [collection_names]
        
        try:
            # Check cache
            if self._query_cache is not None:
                cache_key = self._get_cache_key(user_question, collection_names)
                if cache_key in self._query_cache:
                    logger.info("Returning cached result")
                    return self._query_cache[cache_key]
            
            logger.info(f"Processing query: '{user_question[:50]}...' across {len(collection_names)} collection(s)")
            
            # Retrieve documents
            all_docs, source_stats = self._retrieve_documents(
                user_question, 
                collection_names,
                k
            )
            
            if not all_docs:
                return {
                    "answer": self._generate_no_results_message(collection_names),
                    "sources": []
                }
            
            # Format context
            context_text, source_stats = self._format_context(all_docs, collection_names)
            
            logger.info(f"Context prepared: {len(all_docs)} chunks from {len(source_stats)} sources")
            
            # Generate response
            answer = self._generate_answer(context_text, user_question)
            
            result = {
                "answer": answer,
                "sources": all_docs,
                "source_stats": source_stats
            }
            
            # Cache result
            if self._query_cache is not None:
                cache_key = self._get_cache_key(user_question, collection_names)
                self._query_cache[cache_key] = result
            
            logger.info("Query completed successfully")
            return result
            
        except VectorStoreError as e:
            logger.error(f"Vector store error during query: {e}")
            return {
                "answer": f"Error accessing document database: {str(e)}",
                "sources": []
            }
        except Exception as e:
            logger.error(f"Unexpected error during query: {e}", exc_info=True)
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": []
            }
    
    def _retrieve_documents(
        self, 
        query: str, 
        collection_names: List[str],
        k: Optional[int] = None
    ) -> tuple[List[Document], Dict]:
        """
        Retrieve documents from collections with error handling
        
        Args:
            query: Search query
            collection_names: Collections to search
            k: Number of results per collection
            
        Returns:
            Tuple of (documents, source_stats)
        """
        k = k or config.RETRIEVAL_K
        all_docs = []
        source_stats = {}
        
        # Single collection - simple retrieval
        if len(collection_names) == 1:
            try:
                retriever = self.vector_store.get_retriever(collection_names[0], k=k)
                docs = retriever.invoke(query)
                logger.debug(f"Retrieved {len(docs)} docs from '{collection_names[0]}'")
                return docs, {collection_names[0]: len(docs)}
            except VectorStoreError as e:
                logger.error(f"Retrieval failed for '{collection_names[0]}': {e}")
                return [], {}
        
        # Multiple collections - use multi-collection retriever
        try:
            multi_retriever = self.vector_store.get_multi_collection_retriever(
                collection_names, 
                k=k
            )
            all_docs = multi_retriever.invoke(query)
            
            # Calculate source statistics
            for doc in all_docs:
                source_file = Path(doc.metadata.get("source", "Unknown")).name
                source_stats[source_file] = source_stats.get(source_file, 0) + 1
            
            logger.info(f"Multi-collection retrieval: {len(all_docs)} docs from {len(source_stats)} sources")
            return all_docs, source_stats
            
        except VectorStoreError as e:
            logger.error(f"Multi-collection retrieval failed: {e}")
            return [], {}
    
    def _generate_answer(self, context: str, question: str) -> str:
        """
        Generate answer using LLM with error handling
        
        Args:
            context: Formatted context text
            question: User question
            
        Returns:
            Generated answer
        """
        try:
            chain = self.prompt | self.llm
            
            logger.debug(f"Generating answer (context length: {len(context)} chars)")
            response = chain.invoke({
                "context": context, 
                "question": question
            })
            
            # Clean up response
            if isinstance(response, str):
                response = response.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            return (
                f"I encountered an error while generating the response: {str(e)}\n\n"
                "Please try rephrasing your question or check if the Ollama service is running."
            )
    
    def _generate_no_results_message(self, collection_names: List[str]) -> str:
        """
        Generate helpful message when no results found
        
        Args:
            collection_names: Collections that were searched
            
        Returns:
            Helpful error message
        """
        docs_str = ', '.join(collection_names)
        
        return f"""I couldn't find relevant information in the selected document(s): {docs_str}

Suggestions:
• Try rephrasing your question with different keywords
• Check if the information exists in these specific documents
• For financial metrics, try terms like: revenue, profit, expenses, assets, liabilities
• For comparisons, ensure you've selected multiple documents

Example questions that work well:
- "What was the total revenue?"
- "Compare the expenses between these reports"
- "What are the main risk factors?"
- "Show me the cash flow from operations"
"""
    
    def clear_cache(self):
        """Clear the query cache"""
        if self._query_cache is not None:
            self._query_cache.clear()
            logger.info("Query cache cleared")
    
    def get_cache_size(self) -> int:
        """Get number of cached queries"""
        if self._query_cache is not None:
            return len(self._query_cache)
        return 0
    
    def get_system_info(self) -> Dict:
        """
        Get system information for debugging
        
        Returns:
            Dictionary with system info
        """
        try:
            collections = self.vector_store.list_collections()
            total_docs = sum(c['count'] for c in collections)
            
            return {
                'llm_model': config.LLM_MODEL,
                'embedding_model': config.EMBEDDING_MODEL,
                'num_collections': len(collections),
                'total_documents': total_docs,
                'cache_size': self.get_cache_size(),
                'cache_enabled': config.CACHE_ENABLED,
                'retrieval_k': config.RETRIEVAL_K
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {'error': str(e)}


# ===== Testing & Examples =====

if __name__ == "__main__":
    print("Testing RAG Engine...")
    
    try:
        # Initialize engine
        engine = FinancialRAGEngine()
        print("✅ Engine initialized")
        
        # Show system info
        info = engine.get_system_info()
        print("\n📊 System Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test query (requires indexed documents)
        print("\n--- Testing Query ---")
        
        # This assumes you have at least one collection
        collections = engine.vector_store.list_collections()
        if collections:
            test_collection = collections[0]['name']
            print(f"Testing with collection: {test_collection}")
            
            result = engine.query(
                "What are the main highlights?",
                collection_names=[test_collection]
            )
            
            print(f"\n✅ Query completed")
            print(f"Answer length: {len(result['answer'])} chars")
            print(f"Sources: {len(result['sources'])} documents")
            
            if result.get('source_stats'):
                print("\nSource breakdown:")
                for source, count in result['source_stats'].items():
                    print(f"  - {source}: {count} chunks")
            
            print(f"\nAnswer preview:")
            print(result['answer'][:300] + "...")
            
        else:
            print("⚠️  No collections found. Please index some documents first.")
        
        # Test cache
        if config.CACHE_ENABLED:
            print(f"\n📦 Cache size: {engine.get_cache_size()} queries")
        
    except RAGEngineError as e:
        print(f"❌ Engine error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()