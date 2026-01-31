"""
RAGAS Evaluator Module

Implements the "Truth Meter" feature using RAGAS framework to evaluate
RAG system performance with metrics like Faithfulness, Answer Relevance,
Context Precision, and Context Recall.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Data class to store evaluation results"""
    query: str
    answer: str
    contexts: List[str]
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    context_relevancy: Optional[float] = None
    timestamp: datetime = None
    collection_names: List[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.collection_names is None:
            self.collection_names = []


class RAGASEvaluator:
    """RAGAS-based evaluator for RAG system performance"""
    
    def __init__(self, llm: BaseLanguageModel):
        """
        Initialize the RAGAS evaluator
        
        Args:
            llm: Language model for evaluation
        """
        self.llm = llm
        self.evaluation_history: List[EvaluationResult] = []
        
        # Configure RAGAS metrics
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        
        logger.info("RAGAS evaluator initialized")
    
    def evaluate_query(
        self, 
        query: str, 
        answer: str, 
        retrieved_docs: List[Document],
        collection_names: List[str]
    ) -> EvaluationResult:
        """
        Evaluate a single query-response pair using RAGAS metrics
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_docs: Retrieved documents used for generation
            collection_names: Names of collections used
            
        Returns:
            EvaluationResult with all metrics
        """
        try:
            # Extract contexts from documents
            contexts = [doc.page_content for doc in retrieved_docs]
            
            # Prepare data for RAGAS evaluation
            data = {
                'question': [query],
                'answer': [answer],
                'contexts': [contexts]
            }
            
            # Create evaluation dataset
            from datasets import Dataset
            dataset = Dataset.from_dict(data)
            
            # Evaluate using RAGAS with comprehensive error handling
            try:
                result = evaluate(
                    dataset=dataset,
                    metrics=self.metrics,
                    llm=self.llm
                )
                
                # Extract results
                eval_result = EvaluationResult(
                    query=query,
                    answer=answer,
                    contexts=contexts,
                    faithfulness=result.get('faithfulness'),
                    answer_relevancy=result.get('answer_relevancy'),
                    context_precision=result.get('context_precision'),
                    context_recall=result.get('context_recall'),
                    collection_names=collection_names
                )
                
                logger.info(f"Evaluation completed: Faithfulness={eval_result.faithfulness}")
                return eval_result
                
            except ValueError as ve:
                if "reference" in str(ve):
                    logger.warning("RAGAS evaluation skipped due to missing reference column requirement")
                    # Create evaluation result with None values for unsupported metrics
                    eval_result = EvaluationResult(
                        query=query,
                        answer=answer,
                        contexts=contexts,
                        collection_names=collection_names
                    )
                    return eval_result
                else:
                    raise ve
            except Exception as eval_error:
                logger.warning(f"RAGAS evaluation failed with error: {eval_error}")
                # Create evaluation result with None values for unsupported metrics
                eval_result = EvaluationResult(
                    query=query,
                    answer=answer,
                    contexts=contexts,
                    collection_names=collection_names
                )
                return eval_result
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
            return self._create_failed_evaluation(query, answer, contexts, collection_names, str(e))
    
    def _create_failed_evaluation(
        self, 
        query: str, 
        answer: str, 
        contexts: List[str],
        collection_names: List[str],
        error_msg: str
    ) -> EvaluationResult:
        """Create evaluation result when RAGAS evaluation fails"""
        return EvaluationResult(
            query=query,
            answer=answer,
            contexts=contexts,
            collection_names=collection_names
        )
    
    def get_security_dashboard_data(self) -> Dict[str, Any]:
        """
        Get aggregated security dashboard data
        
        Returns:
            Dictionary with aggregated metrics for dashboard
        """
        if not self.evaluation_history:
            return {
                'total_evaluations': 0,
                'avg_faithfulness': 0,
                'avg_answer_relevancy': 0,
                'avg_context_precision': 0,
                'avg_context_recall': 0,
                'avg_context_relevancy': 0,
                'recent_evaluations': [],
                'trend_data': {}
            }
        
        # Calculate averages
        valid_evals = [e for e in self.evaluation_history if e.faithfulness is not None]
        
        if not valid_evals:
            return {
                'total_evaluations': len(self.evaluation_history),
                'avg_faithfulness': 0,
                'avg_answer_relevancy': 0,
                'avg_context_precision': 0,
                'avg_context_recall': 0,
                'avg_context_relevancy': 0,
                'recent_evaluations': self.evaluation_history[-5:],
                'trend_data': {}
            }
        
        # Calculate metrics
        metrics = {
            'total_evaluations': len(self.evaluation_history),
            'avg_faithfulness': np.mean([e.faithfulness for e in valid_evals if e.faithfulness is not None]),
            'avg_answer_relevancy': np.mean([e.answer_relevancy for e in valid_evals if e.answer_relevancy is not None]),
            'avg_context_precision': np.mean([e.context_precision for e in valid_evals if e.context_precision is not None]),
            'avg_context_recall': np.mean([e.context_recall for e in valid_evals if e.context_recall is not None]),
            'avg_context_relevancy': np.mean([e.context_relevancy for e in valid_evals if e.context_relevancy is not None]),
        }
        
        # Recent evaluations
        metrics['recent_evaluations'] = self.evaluation_history[-5:]
        
        # Trend data for charts
        metrics['trend_data'] = self._calculate_trends(valid_evals)
        
        return metrics
    
    def _calculate_trends(self, evaluations: List[EvaluationResult]) -> Dict[str, List[float]]:
        """Calculate trend data for dashboard charts"""
        if len(evaluations) < 2:
            return {}
        
        # Sort by timestamp
        sorted_evals = sorted(evaluations, key=lambda x: x.timestamp)
        
        # Calculate rolling averages (window of 5)
        window_size = min(5, len(sorted_evals))
        
        trends = {}
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
            values = [getattr(e, metric) for e in sorted_evals if getattr(e, metric) is not None]
            if len(values) >= window_size:
                # Calculate rolling average
                rolling_avg = []
                for i in range(len(values)):
                    start_idx = max(0, i - window_size + 1)
                    window = values[start_idx:i+1]
                    rolling_avg.append(np.mean(window))
                trends[metric] = rolling_avg
        
        return trends
    
    def get_collection_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics by collection
        
        Returns:
            Dictionary with collection-wise performance metrics
        """
        if not self.evaluation_history:
            return {}
        
        # Group by collection
        collection_metrics = {}
        for eval_result in self.evaluation_history:
            for collection in eval_result.collection_names:
                if collection not in collection_metrics:
                    collection_metrics[collection] = {
                        'faithfulness': [],
                        'answer_relevancy': [],
                        'context_precision': [],
                        'context_recall': [],
                        'count': 0
                    }
                
                # Add metrics if available
                if eval_result.faithfulness is not None:
                    collection_metrics[collection]['faithfulness'].append(eval_result.faithfulness)
                if eval_result.answer_relevancy is not None:
                    collection_metrics[collection]['answer_relevancy'].append(eval_result.answer_relevancy)
                if eval_result.context_precision is not None:
                    collection_metrics[collection]['context_precision'].append(eval_result.context_precision)
                if eval_result.context_recall is not None:
                    collection_metrics[collection]['context_recall'].append(eval_result.context_recall)
                
                collection_metrics[collection]['count'] += 1
        
        # Calculate averages
        for collection in collection_metrics:
            for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                values = collection_metrics[collection][metric]
                if values:
                    collection_metrics[collection][metric] = np.mean(values)
                else:
                    collection_metrics[collection][metric] = 0
        
        return collection_metrics
    
    def export_evaluation_report(self, filename: str = None) -> str:
        """
        Export evaluation history to CSV file
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        if not self.evaluation_history:
            logger.warning("No evaluation history to export")
            return ""
        
        # Prepare data for export
        export_data = []
        for eval_result in self.evaluation_history:
            export_data.append({
                'timestamp': eval_result.timestamp.isoformat(),
                'query': eval_result.query,
                'answer': eval_result.answer,
                'collections': ', '.join(eval_result.collection_names),
                'faithfulness': eval_result.faithfulness,
                'answer_relevancy': eval_result.answer_relevancy,
                'context_precision': eval_result.context_precision,
                'context_recall': eval_result.context_recall,
                'context_relevancy': eval_result.context_relevancy,
                'num_contexts': len(eval_result.contexts)
            })
        
        # Create DataFrame and export
        df = pd.DataFrame(export_data)
        
        if filename is None:
            filename = f"ragas_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = f"data/evaluations/{filename}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Evaluation report exported to: {filepath}")
        return filepath
    
    def clear_history(self):
        """Clear evaluation history"""
        self.evaluation_history.clear()
        logger.info("Evaluation history cleared")


def create_security_dashboard(evaluator: RAGASEvaluator):
    """
    Create Streamlit security dashboard component
    
    Args:
        evaluator: RAGAS evaluator instance
    """
    
    # Get dashboard data
    dashboard_data = evaluator.get_security_dashboard_data()
    
    if dashboard_data['total_evaluations'] == 0:
        st.info("""
        **No evaluations available yet.** 
        
        To see performance metrics, first ask questions in the main chat interface. 
        The Security Dashboard will automatically evaluate each query and show:
        
        - **Faithfulness**: How often AI answers are supported by retrieved context
        - **Answer Relevance**: How well answers address your questions  
        - **Context Precision**: How relevant the retrieved information is
        - **Context Recall**: How much relevant information was found
        
        **Try asking:**
        - "What was Tesla's revenue in 2023?"
        - "Compare Apple and Tesla's profit margins"
        - "What are the main risk factors mentioned?"
        """)
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Evaluations",
            dashboard_data['total_evaluations'],
            help="Total number of queries evaluated"
        )
    
    with col2:
        faithfulness_score = dashboard_data['avg_faithfulness']
        st.metric(
            "Faithfulness",
            f"{faithfulness_score:.1%}",
            help="How often the AI answers are supported by retrieved context (lower hallucination rate)",
            delta_color="normal"
        )
    
    with col3:
        relevancy_score = dashboard_data['avg_answer_relevancy']
        st.metric(
            "Answer Relevance",
            f"{relevancy_score:.1%}",
            help="How well answers address the user's question",
            delta_color="normal"
        )
    
    with col4:
        precision_score = dashboard_data['avg_context_precision']
        st.metric(
            "Context Precision",
            f"{precision_score:.1%}",
            help="How relevant the retrieved context chunks are",
            delta_color="normal"
        )
    
    # Additional metrics row
    col5, col6 = st.columns(2)
    
    with col5:
        recall_score = dashboard_data['avg_context_recall']
        st.metric(
            "Context Recall",
            f"{recall_score:.1%}",
            help="How much relevant information was retrieved",
            delta_color="normal"
        )
    
    with col6:
        context_relevancy_score = dashboard_data['avg_context_relevancy']
        st.metric(
            "Context Relevancy",
            f"{context_relevancy_score:.1%}",
            help="Overall quality of retrieved context",
            delta_color="normal"
        )
    
    # Performance trends chart
    if dashboard_data['trend_data']:
        st.subheader("📈 Performance Trends")
        
        # Create trend chart
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Faithfulness Trend', 'Answer Relevance Trend', 
                          'Context Precision Trend', 'Context Recall Trend'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        metrics_to_plot = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in dashboard_data['trend_data']:
                values = dashboard_data['trend_data'][metric]
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                fig.add_trace(
                    go.Scatter(
                        y=values,
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(color=colors[i], width=2),
                        marker=dict(size=4)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text="RAG Performance Trends (Rolling Average)",
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Collection performance
    collection_performance = evaluator.get_collection_performance()
    if collection_performance:
        st.subheader("📊 Collection Performance")
        
        col_data = []
        for collection, metrics in collection_performance.items():
            col_data.append({
                'Collection': collection,
                'Faithfulness': f"{metrics['faithfulness']:.1%}",
                'Answer Relevance': f"{metrics['answer_relevancy']:.1%}",
                'Context Precision': f"{metrics['context_precision']:.1%}",
                'Context Recall': f"{metrics['context_recall']:.1%}",
                'Queries': metrics['count']
            })
        
        st.dataframe(col_data, use_container_width=True)
    
    # Recent evaluations
    if dashboard_data['recent_evaluations']:
        st.subheader("📋 Recent Evaluations")
        
        for i, eval_result in enumerate(dashboard_data['recent_evaluations'][-3:]):
            with st.expander(f"Query {i+1}: {eval_result.query[:50]}..."):
                st.write(f"**Collections:** {', '.join(eval_result.collection_names)}")
                st.write(f"**Faithfulness:** {eval_result.faithfulness:.3f}" if eval_result.faithfulness else "**Faithfulness:** N/A")
                st.write(f"**Answer Relevance:** {eval_result.answer_relevancy:.3f}" if eval_result.answer_relevancy else "**Answer Relevance:** N/A")
                st.write(f"**Context Precision:** {eval_result.context_precision:.3f}" if eval_result.context_precision else "**Context Precision:** N/A")
                st.write(f"**Context Recall:** {eval_result.context_recall:.3f}" if eval_result.context_recall else "**Context Recall:** N/A")
                
                st.write("**Answer:**")
                st.text_area(f"Answer_{i}", eval_result.answer, height=100, key=f"answer_{i}")
    
    # Export options
    st.divider()
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🗑️ Clear History", type="secondary"):
            evaluator.clear_history()
            st.rerun()
    
    with col2:
        if st.button("📊 Export Report", type="primary"):
            filepath = evaluator.export_evaluation_report()
            if filepath:
                st.success(f"Report exported to: {filepath}")
            else:
                st.error("Failed to export report")