"""
Advanced Features Integration Module

Integrates RAGAS evaluation, GraphRAG, and LangGraph agent features
into the main RAG engine for a comprehensive advanced RAG system.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
from pathlib import Path

from src.evaluation.ragas_evaluator import RAGASEvaluator, create_security_dashboard
from src.graph.graph_rag import GraphRAG, create_graph_visualization
from src.agent.agent_orchestrator import FinancialAgent, create_agent_interface
from src.engine import FinancialRAGEngine
from src.utils.vector_store import FinancialVectorStore
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


class AdvancedRAGSystem:
    """Main integration class for all advanced RAG features"""
    
    def __init__(self, base_engine: FinancialRAGEngine):
        """
        Initialize the advanced RAG system
        
        Args:
            base_engine: Base RAG engine instance
        """
        self.base_engine = base_engine
        self.vector_store = base_engine.vector_store
        self.llm = base_engine.llm
        
        # Initialize advanced features
        self.evaluator = RAGASEvaluator(self.llm)
        self.graph_rag = GraphRAG(self.llm)
        self.agent = FinancialAgent(self.llm, self.vector_store)
        
        # Feature states
        self.features_enabled = {
            'evaluation': True,
            'graph_rag': True,
            'agent': True
        }
        
        # Cache for processed documents
        self.processed_documents: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Advanced RAG system initialized with all features")
    
    def query_with_evaluation(
        self, 
        user_question: str, 
        collection_names: List[str],
        enable_evaluation: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system with optional RAGAS evaluation
        
        Args:
            user_question: User's question
            collection_names: Collections to query
            enable_evaluation: Whether to run RAGAS evaluation
            
        Returns:
            Query result with optional evaluation
        """
        # Run base query
        result = self.base_engine.query(user_question, collection_names)
        
        # Add evaluation if enabled
        if enable_evaluation and self.features_enabled['evaluation']:
            try:
                evaluation_result = self.evaluator.evaluate_query(
                    user_question,
                    result['answer'],
                    result['sources'],
                    collection_names
                )
                
                result['evaluation'] = {
                    'faithfulness': evaluation_result.faithfulness,
                    'answer_relevancy': evaluation_result.answer_relevancy,
                    'context_precision': evaluation_result.context_precision,
                    'context_recall': evaluation_result.context_recall,
                    'context_relevancy': evaluation_result.context_relevancy
                }
                
                # Add evaluation metrics to answer
                if all(v is not None for v in result['evaluation'].values()):
                    metrics_summary = self._format_evaluation_summary(result['evaluation'])
                    result['answer'] += f"\n\n📊 **Evaluation Metrics:** {metrics_summary}"
                
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                result['evaluation'] = {'error': str(e)}
        
        return result
    
    def _format_evaluation_summary(self, evaluation: Dict[str, float]) -> str:
        """Format evaluation metrics for display"""
        summary_parts = []
        for metric, value in evaluation.items():
            if value is not None:
                metric_name = metric.replace('_', ' ').title()
                summary_parts.append(f"{metric_name}: {value:.1%}")
        
        return " | ".join(summary_parts)
    
    def build_knowledge_graph(
        self, 
        collection_names: List[str],
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from documents
        
        Args:
            collection_names: Collections to process
            force_rebuild: Whether to force rebuild the graph
            
        Returns:
            Graph building results
        """
        if not self.features_enabled['graph_rag']:
            return {'error': 'GraphRAG feature is disabled'}
        
        try:
            # Check if already processed
            cache_key = "_".join(sorted(collection_names))
            if cache_key in self.processed_documents and not force_rebuild:
                logger.info(f"Using cached graph for {collection_names}")
                return {'status': 'cached', 'message': 'Using cached graph data'}
            
            # Retrieve documents
            all_docs = []
            for collection_name in collection_names:
                try:
                    retriever = self.vector_store.get_retriever(collection_name, k=100)
                    docs = retriever.invoke("financial information")
                    all_docs.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to retrieve from {collection_name}: {e}")
            
            if not all_docs:
                return {'error': 'No documents found for graph building'}
            
            # Extract entities
            documents = [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in all_docs
            ]
            
            entities = self.graph_rag.extract_entities_from_documents(documents)
            
            # Extract relationships using fast pattern-based method
            relationships = self.graph_rag.extract_relationships_fast(entities, documents)
            
            # Build graph
            self.graph_rag.build_knowledge_graph(entities, relationships)
            
            # Cache results
            self.processed_documents[cache_key] = {
                'entities': entities,
                'relationships': relationships,
                'timestamp': datetime.now()
            }
            
            stats = self.graph_rag.get_graph_statistics()
            
            return {
                'status': 'success',
                'entities_extracted': len(entities),
                'relationships_found': len(relationships),
                'graph_stats': stats,
                'message': f'Knowledge graph built with {len(entities)} entities and {len(relationships)} relationships'
            }
            
        except Exception as e:
            logger.error(f"Graph building failed: {e}", exc_info=True)
            return {'error': str(e)}
    
    def run_agent_analysis(
        self, 
        user_query: str, 
        collection_names: List[str]
    ) -> Dict[str, Any]:
        """
        Run multi-step agent analysis
        
        Args:
            user_query: User's analysis request
            collection_names: Collections to analyze
            
        Returns:
            Agent analysis results
        """
        if not self.features_enabled['agent']:
            return {'error': 'Agent feature is disabled'}
        
        try:
            result = asyncio.run(
                self.agent.run_analysis(user_query, collection_names)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Agent analysis failed: {e}", exc_info=True)
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all advanced features"""
        status = {
            'base_engine': {
                'initialized': self.base_engine is not None,
                'llm_model': getattr(self.llm, 'model', 'Unknown') if self.llm else 'None',
                'collections': len(self.vector_store.list_collections())
            },
            'evaluation': {
                'enabled': self.features_enabled['evaluation'],
                'evaluations_count': len(self.evaluator.evaluation_history),
                'last_evaluation': self._get_last_evaluation_time()
            },
            'graph_rag': {
                'enabled': self.features_enabled['graph_rag'],
                'graphs_built': len(self.processed_documents),
                'total_entities': sum(len(data.get('entities', [])) for data in self.processed_documents.values()),
                'total_relationships': sum(len(data.get('relationships', [])) for data in self.processed_documents.values())
            },
            'agent': {
                'enabled': self.features_enabled['agent'],
                'tools_available': len(self.agent.tools),
                'tool_names': [tool.name for tool in self.agent.tools]
            }
        }
        
        return status
    
    def _get_last_evaluation_time(self) -> Optional[str]:
        """Get timestamp of last evaluation"""
        if self.evaluator.evaluation_history:
            return self.evaluator.evaluation_history[-1].timestamp.isoformat()
        return None
    
    def export_all_reports(self, output_dir: str = "data/reports") -> Dict[str, str]:
        """
        Export reports from all features
        
        Args:
            output_dir: Output directory for reports
            
        Returns:
            Paths to exported files
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        exported_files = {}
        
        # Export RAGAS evaluation report
        if self.features_enabled['evaluation']:
            try:
                filepath = self.evaluator.export_evaluation_report()
                if filepath:
                    exported_files['evaluation_report'] = filepath
            except Exception as e:
                logger.error(f"Failed to export evaluation report: {e}")
        
        # Export graph data
        if self.features_enabled['graph_rag'] and self.graph_rag.graph:
            try:
                filepath = self.graph_rag.export_graph_data()
                if filepath:
                    exported_files['graph_data'] = filepath
            except Exception as e:
                logger.error(f"Failed to export graph data: {e}")
        
        return exported_files
    
    def clear_all_caches(self):
        """Clear all feature caches"""
        # Clear evaluation cache
        self.evaluator.clear_history()
        
        # Clear graph cache
        self.graph_rag.graph.clear()
        self.graph_rag.entities.clear()
        self.graph_rag.relationships.clear()
        self.processed_documents.clear()
        
        # Clear base engine cache
        self.base_engine.clear_cache()
        
        logger.info("All caches cleared")


def create_advanced_features_ui(advanced_system: AdvancedRAGSystem, collections: List[str]):
    """
    Create Streamlit UI for all advanced features
    
    Args:
        advanced_system: AdvancedRAGSystem instance
        collections: Available collections
    """
    st.title("🚀 Advanced RAG Features")
    
    st.write("""
    **Welcome to the advanced features of Sentinel Financial AI!** 
    These features provide cutting-edge capabilities for financial document analysis:
    """)
    
    # Feature overview cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🛡️ Security Dashboard")
        st.write("**Purpose:** Measure AI truthfulness and accuracy")
        st.write("**Key Metrics:** Faithfulness, Answer Relevance, Context Precision")
        st.write("**Use Cases:** Quality assurance, Performance monitoring")
        st.write("**LinkedIn Value:** Shows AI self-grading capabilities")
    
    with col2:
        st.subheader("🕸️ Knowledge Graph")
        st.write("**Purpose:** Visualize relationships between entities")
        st.write("**Key Metrics:** Entities, Relationships, Connected Components")
        st.write("**Use Cases:** Relationship mapping, Network analysis")
        st.write("**LinkedIn Value:** Visual knowledge graphs are impressive")
    
    with col3:
        st.subheader("🤖 Multi-Step Agent")
        st.write("**Purpose:** Perform multi-step analytical workflows")
        st.write("**Key Metrics:** Analysis Steps, Tool Usage, Report Quality")
        st.write("**Use Cases:** Complex analysis, Automated workflows")
        st.write("**LinkedIn Value:** Demonstrates advanced AI reasoning")
    
    # System status
    status = advanced_system.get_system_status()
    
    st.divider()
    st.subheader("📊 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "RAGAS Evaluations",
            status['evaluation']['evaluations_count'],
            help="Number of queries evaluated for quality"
        )
    
    with col2:
        st.metric(
            "Knowledge Graph Entities",
            status['graph_rag']['total_entities'],
            help="Total entities extracted from documents"
        )
    
    with col3:
        st.metric(
            "Agent Tools",
            status['agent']['tools_available'],
            help="Available tools for multi-step analysis"
        )
    
    # Feature tabs
    tab1, tab2, tab3 = st.tabs(["🛡️ Security Dashboard", "🕸️ Knowledge Graph", "🤖 Multi-Step Agent"])
    
    with tab1:
        st.header("🛡️ RAGAS Security Dashboard")
        create_security_dashboard(advanced_system.evaluator)
    
    with tab2:
        st.header("🕸️ Knowledge Graph Visualization")
        
        # Compact, elegant button layout at the top
        st.markdown("### 🎛️ Graph Controls")
        
        # Create a more compact button layout
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        
        with col1:
            # Main build button - smaller and more elegant
            if st.button("🚀 Build Graph", type="primary", use_container_width=True):
                with st.spinner("Building knowledge graph... This may take 5-10 minutes as we process documents and extract entities..."):
                    result = advanced_system.build_knowledge_graph(collections)
                    
                    if 'error' in result:
                        st.error(f"Failed to build graph: {result['error']}")
                    else:
                        st.success(result['message'])
                        st.rerun()
        
        with col2:
            # Quick build with selection
            selected_collections = st.multiselect(
                "Collections:",
                options=collections,
                default=collections[:2] if len(collections) >= 2 else collections,
                key="graph_collections"
            )
            
            if st.button("⚡ Quick Build", type="secondary", use_container_width=True):
                if selected_collections:
                    with st.spinner("Building knowledge graph..."):
                        result = advanced_system.build_knowledge_graph(selected_collections)
                        
                        if 'error' in result:
                            st.error(f"Failed to build graph: {result['error']}")
                        else:
                            st.success(result['message'])
                            st.rerun()
                else:
                    st.error("Please select at least one collection")
        
        with col3:
            # Clear button - smaller
            if st.button("🗑️ Clear", type="secondary", use_container_width=True):
                advanced_system.graph_rag.graph.clear()
                advanced_system.graph_rag.entities.clear()
                advanced_system.graph_rag.relationships.clear()
                st.rerun()
        
        with col4:
            # Stats button - compact
            if st.button("📊 Stats", type="secondary", use_container_width=True):
                stats = advanced_system.graph_rag.get_graph_statistics()
                if stats:
                    st.info(f"""
                    **Graph Statistics:**
                    - Nodes: {stats.get('nodes', 0)}
                    - Edges: {stats.get('edges', 0)}
                    - Components: {stats.get('components', 0)}
                    """)
                else:
                    st.info("No graph data available")
        
        # Show warning if no graph built yet
        if not advanced_system.graph_rag.graph or len(advanced_system.graph_rag.graph.nodes) == 0:
            st.warning("""
            **No knowledge graph built yet!**
            
            **Quick Start:**
            1. Documents are already activated: """ + ", ".join(collections) + """
            2. Click the **"Build Knowledge Graph"** button above
            3. Wait for processing to complete
            4. View the interactive graph visualization
            """)
        
        st.info("""
        **How to Build a Knowledge Graph:**
        
        1. **Select documents** from the sidebar and click "Activate Selection"
        2. **Choose collections** above for graph building
        3. **Click "Build Knowledge Graph"** to process documents
        4. **Wait for processing** - system extracts entities and relationships
        5. **View interactive graph** once processing completes
        
        **What you'll see:**
        - 🔴 Companies, 🔵 People, 🟢 Locations, 🟡 Financial Metrics
        - Lines connecting related entities
        - Click nodes for detailed information
        """)
        
        # Graph visualization takes the full width
        create_graph_visualization(advanced_system.graph_rag)
    
    with tab3:
        st.header("🤖 Multi-Step Agent Analysis")
        create_agent_interface(advanced_system.agent, collections)
    
    # System management
    st.divider()
    st.subheader("⚙️ System Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export All Reports", type="primary"):
            with st.spinner("Exporting reports..."):
                exported = advanced_system.export_all_reports()
                if exported:
                    st.success("Reports exported successfully!")
                    for report_type, filepath in exported.items():
                        st.write(f"- {report_type}: {filepath}")
                else:
                    st.info("No reports to export")
    
    with col2:
        if st.button("🧹 Clear All Caches", type="secondary"):
            advanced_system.clear_all_caches()
            st.success("All caches cleared!")
            st.rerun()
    
    with col3:
        if st.button("ℹ️ System Status"):
            with st.expander("System Information"):
                st.json(status)


def create_feature_comparison():
    """Create a comparison table of the three advanced features"""
    import streamlit as st
    st.subheader("📋 Feature Comparison")
    
    features_data = {
        'Feature': ['RAGAS Evaluation', 'GraphRAG', 'LangGraph Agent'],
        'Purpose': [
            'Measure AI truthfulness and accuracy',
            'Visualize relationships between entities',
            'Perform multi-step analytical workflows'
        ],
        'Key Metrics': [
            'Faithfulness, Answer Relevance, Context Precision',
            'Entities, Relationships, Connected Components',
            'Analysis Steps, Tool Usage, Report Quality'
        ],
        'Use Cases': [
            'Quality assurance, Performance monitoring',
            'Relationship mapping, Network analysis',
            'Complex analysis, Automated workflows'
        ],
        'LinkedIn Value': [
            'Shows AI self-grading capabilities',
            'Visual knowledge graphs are impressive',
            'Demonstrates advanced AI reasoning'
        ]
    }
    
    import pandas as pd
    df = pd.DataFrame(features_data)
    st.dataframe(df, use_container_width=True)