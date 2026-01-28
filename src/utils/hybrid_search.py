"""
Hybrid search implementation combining semantic and keyword-based retrieval.
Provides advanced search capabilities for financial documents with multimodal support.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    document: Document
    score: float
    search_type: str  # 'semantic', 'keyword', or 'hybrid'
    metadata: Dict[str, Any]


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining semantic and keyword search."""
    
    vector_store: Chroma
    embeddings: Embeddings
    keyword_weight: float = 0.3
    semantic_weight: float = 0.7
    top_k: int = 10
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get relevant documents using hybrid search."""
        try:
            # Perform semantic search
            semantic_results = self._semantic_search(query)
            
            # Perform keyword search
            keyword_results = self._keyword_search(query)
            
            # Combine results with hybrid scoring
            combined_results = self._combine_results(semantic_results, keyword_results, query)
            
            # Return top documents
            return [result.document for result in combined_results[:self.top_k]]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fall back to semantic search only
            return self._semantic_search(query)
    
    def _semantic_search(self, query: str) -> List[SearchResult]:
        """Perform semantic search using vector embeddings."""
        try:
            # Get semantic results from vector store
            docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(
                query, k=self.top_k * 2
            )
            
            results = []
            for doc, score in docs_and_scores:
                results.append(SearchResult(
                    document=doc,
                    score=score,
                    search_type='semantic',
                    metadata={'semantic_score': score}
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _keyword_search(self, query: str) -> List[SearchResult]:
        """Perform keyword-based search using text matching."""
        try:
            # Get all documents from vector store
            all_docs = self.vector_store.get()
            
            if not all_docs or 'documents' not in all_docs:
                return []
            
            results = []
            query_tokens = self._tokenize_query(query)
            
            for doc in all_docs['documents']:
                # Calculate keyword match score
                score = self._calculate_keyword_score(query_tokens, doc.page_content)
                
                if score > 0:
                    results.append(SearchResult(
                        document=doc,
                        score=score,
                        search_type='keyword',
                        metadata={'keyword_score': score}
                    ))
            
            # Sort by score descending
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:self.top_k * 2]
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize query into meaningful terms."""
        # Convert to lowercase and remove punctuation
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        
        # Split into tokens and filter common words
        tokens = query.split()
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should'
        }
        
        return [token for token in tokens if token not in stop_words and len(token) > 2]
    
    def _calculate_keyword_score(self, query_tokens: List[str], text: str) -> float:
        """Calculate keyword match score between query and document."""
        if not query_tokens:
            return 0.0
        
        # Convert text to lowercase for matching
        text_lower = text.lower()
        
        # Count matches
        matches = 0
        for token in query_tokens:
            if token in text_lower:
                matches += 1
        
        # Calculate normalized score
        score = matches / len(query_tokens)
        
        # Boost score for financial terms
        financial_boost = self._calculate_financial_boost(query_tokens, text_lower)
        
        return score * (1 + financial_boost)
    
    def _calculate_financial_boost(self, query_tokens: List[str], text: str) -> float:
        """Calculate boost for financial terminology matches."""
        financial_terms = {
            'revenue', 'profit', 'loss', 'income', 'expense', 'asset', 'liability',
            'equity', 'cash', 'flow', 'balance', 'sheet', 'statement', 'earnings',
            'eps', 'ebitda', 'margin', 'ratio', 'growth', 'forecast', 'guidance',
            'quarter', 'annual', 'fiscal', 'year', 'q1', 'q2', 'q3', 'q4'
        }
        
        boost = 0.0
        for token in query_tokens:
            if token in financial_terms and token in text:
                boost += 0.1
        
        return min(boost, 0.5)  # Cap boost at 50%
    
    def _combine_results(
        self, 
        semantic_results: List[SearchResult], 
        keyword_results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """Combine semantic and keyword search results."""
        # Create a map of documents to their scores
        doc_scores = {}
        
        # Add semantic scores
        for result in semantic_results:
            doc_id = id(result.document)
            doc_scores[doc_id] = {
                'document': result.document,
                'semantic_score': result.score,
                'keyword_score': 0.0,
                'metadata': result.metadata.copy()
            }
        
        # Add keyword scores
        for result in keyword_results:
            doc_id = id(result.document)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'document': result.document,
                    'semantic_score': 0.0,
                    'keyword_score': result.score,
                    'metadata': result.metadata.copy()
                }
            else:
                doc_scores[doc_id]['keyword_score'] = result.score
                doc_scores[doc_id]['metadata'].update(result.metadata)
        
        # Calculate hybrid scores
        combined_results = []
        for doc_info in doc_scores.values():
            hybrid_score = (
                doc_info['semantic_score'] * self.semantic_weight +
                doc_info['keyword_score'] * self.keyword_weight
            )
            
            # Add multimodal boost if document contains tables/charts
            multimodal_boost = self._calculate_multimodal_boost(doc_info['document'], query)
            hybrid_score *= (1 + multimodal_boost)
            
            combined_results.append(SearchResult(
                document=doc_info['document'],
                score=hybrid_score,
                search_type='hybrid',
                metadata={
                    **doc_info['metadata'],
                    'hybrid_score': hybrid_score,
                    'semantic_score': doc_info['semantic_score'],
                    'keyword_score': doc_info['keyword_score'],
                    'multimodal_boost': multimodal_boost
                }
            ))
        
        # Sort by hybrid score descending
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results
    
    def _calculate_multimodal_boost(self, document: Document, query: str) -> float:
        """Calculate boost for documents with multimodal content relevant to query."""
        try:
            metadata = document.metadata
            
            # Check if document has multimodal content
            if 'multimodal_summary' not in metadata:
                return 0.0
            
            multimodal_info = metadata['multimodal_summary']
            tables_found = multimodal_info.get('tables_found', 0)
            charts_found = multimodal_info.get('charts_found', 0)
            
            if tables_found == 0 and charts_found == 0:
                return 0.0
            
            # Check if query is related to tables or charts
            query_lower = query.lower()
            table_keywords = ['table', 'schedule', 'exhibit', 'data', 'numbers', 'figures']
            chart_keywords = ['chart', 'graph', 'visual', 'trend', 'plot', 'diagram']
            
            table_match = any(keyword in query_lower for keyword in table_keywords)
            chart_match = any(keyword in query_lower for keyword in chart_keywords)
            
            boost = 0.0
            if table_match and tables_found > 0:
                boost += 0.2 * min(tables_found, 3)  # Up to 60% boost for multiple tables
            
            if chart_match and charts_found > 0:
                boost += 0.25 * min(charts_found, 3)  # Up to 75% boost for multiple charts
            
            return min(boost, 1.0)  # Cap boost at 100%
            
        except Exception as e:
            logger.error(f"Error calculating multimodal boost: {e}")
            return 0.0


class AdvancedSearchEngine:
    """Advanced search engine with multimodal and financial domain support."""
    
    def __init__(
        self, 
        vector_store: Chroma, 
        embeddings: Embeddings,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
        top_k: int = 10
    ):
        """
        Initialize the advanced search engine.
        
        Args:
            vector_store: Chroma vector store instance
            embeddings: Embedding model
            keyword_weight: Weight for keyword search (0.0 to 1.0)
            semantic_weight: Weight for semantic search (0.0 to 1.0)
            top_k: Number of top results to return
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.top_k = top_k
        
        # Initialize hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            embeddings=embeddings,
            keyword_weight=keyword_weight,
            semantic_weight=semantic_weight,
            top_k=top_k
        )
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform advanced search with hybrid and multimodal support.
        
        Args:
            query: Search query
            
        Returns:
            List of search results with metadata
        """
        try:
            # Get documents using hybrid retriever
            documents = self.hybrid_retriever.invoke(query)
            
            # Convert to result format
            results = []
            for i, doc in enumerate(documents):
                result = {
                    'rank': i + 1,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'snippet': self._generate_snippet(doc.page_content, query),
                    'relevance_score': doc.metadata.get('hybrid_score', 0.0),
                    'search_type': doc.metadata.get('search_type', 'hybrid'),
                    'multimodal_content': self._extract_multimodal_info(doc.metadata)
                }
                results.append(result)
            
            logger.info(f"Advanced search completed: {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error in advanced search: {e}")
            return []
    
    def _generate_snippet(self, content: str, query: str, max_length: int = 300) -> str:
        """Generate a snippet highlighting query terms in the content."""
        try:
            # Simple snippet generation - find query terms and extract surrounding context
            query_lower = query.lower()
            content_lower = content.lower()
            
            # Find first occurrence of query terms
            for term in query_lower.split():
                if len(term) > 3 and term in content_lower:
                    start = content_lower.find(term)
                    if start != -1:
                        # Extract snippet around the term
                        snippet_start = max(0, start - 100)
                        snippet_end = min(len(content), start + len(term) + 200)
                        
                        snippet = content[snippet_start:snippet_end]
                        if snippet_start > 0:
                            snippet = "..." + snippet
                        if snippet_end < len(content):
                            snippet = snippet + "..."
                        
                        return snippet
            
            # Fallback to first part of content
            return content[:max_length] + "..." if len(content) > max_length else content
            
        except Exception as e:
            logger.error(f"Error generating snippet: {e}")
            return content[:max_length] + "..." if len(content) > max_length else content
    
    def _extract_multimodal_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract multimodal information from document metadata."""
        multimodal_info = metadata.get('multimodal_summary', {})
        
        return {
            'has_tables': multimodal_info.get('tables_found', 0) > 0,
            'has_charts': multimodal_info.get('charts_found', 0) > 0,
            'tables_count': multimodal_info.get('tables_found', 0),
            'charts_count': multimodal_info.get('charts_found', 0),
            'multimodal_processing': multimodal_info.get('multimodal_processing', False)
        }
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search engine."""
        try:
            # Get document count
            doc_count = self.vector_store._collection.count()
            
            return {
                'total_documents': doc_count,
                'search_types': ['semantic', 'keyword', 'hybrid'],
                'multimodal_support': True,
                'keyword_weight': self.keyword_weight,
                'semantic_weight': self.semantic_weight,
                'top_k': self.top_k
            }
            
        except Exception as e:
            logger.error(f"Error getting search statistics: {e}")
            return {'error': str(e)}