"""
LangGraph Agent Orchestrator

Implements the "Agentic Multi-Step Analysis" feature using LangGraph to enable
complex, multi-step analytical workflows with specialized tools for financial analysis.
"""

import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
from datetime import datetime
import asyncio
from functools import partial
import operator

import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema for the agent"""
    messages: Annotated[List[BaseMessage], operator.add]
    current_step: str
    step_results: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    final_answer: Optional[str]
    error: Optional[str]


@dataclass
class AgentTool:
    """Represents a tool available to the agent"""
    name: str
    description: str
    function: callable
    requires_llm: bool = False


class FinancialAgent:
    """LangGraph-based agent for multi-step financial analysis"""
    
    def __init__(self, llm: BaseLanguageModel, vector_store):
        """
        Initialize the financial agent
        
        Args:
            llm: Language model for reasoning and tool calling
            vector_store: Vector store for document search
        """
        self.llm = llm
        self.vector_store = vector_store
        self.tools = self._initialize_tools()
        self.graph = self._build_agent_graph()
        
        logger.info("Financial agent initialized with LangGraph")
    
    def _initialize_tools(self) -> List[AgentTool]:
        """Initialize the agent's toolset"""
        tools = []
        
        # 1. Financial Search Tool
        @tool
        def financial_search(query: str, collections: List[str], k: int = 5) -> Dict[str, Any]:
            """
            Search for financial information across documents
            
            Args:
                query: Search query
                collections: List of collection names to search
                k: Number of results to return
                
            Returns:
                Search results with metadata
            """
            try:
                if isinstance(collections, str):
                    collections = [collections]
                
                retriever = self.vector_store.get_multi_collection_retriever(collections, k=k)
                docs = retriever.invoke(query)
                
                results = []
                for doc in docs:
                    results.append({
                        'content': doc.page_content,
                        'source': doc.metadata.get('source', 'Unknown'),
                        'page': doc.metadata.get('page', 0) + 1,
                        'collection': doc.metadata.get('collection', 'Unknown')
                    })
                
                return {
                    'query': query,
                    'results': results,
                    'total_found': len(results),
                    'collections_searched': collections
                }
                
            except Exception as e:
                logger.error(f"Financial search failed: {e}")
                return {
                    'error': str(e),
                    'query': query,
                    'results': [],
                    'total_found': 0
                }
        
        tools.append(AgentTool(
            name="financial_search",
            description="Search for financial information across indexed documents",
            function=financial_search
        ))
        
        # 2. Calculator Tool
        @tool
        def calculator(operation: str, numbers: List[float]) -> Dict[str, Any]:
            """
            Perform financial calculations
            
            Args:
                operation: Type of calculation ('growth_rate', 'percentage', 'ratio', 'sum', 'average')
                numbers: List of numbers to calculate
                
            Returns:
                Calculation result
            """
            try:
                if not numbers:
                    return {'error': 'No numbers provided'}
                
                if operation == 'growth_rate':
                    if len(numbers) != 2:
                        return {'error': 'Growth rate requires exactly 2 numbers (old_value, new_value)'}
                    old_val, new_val = numbers
                    if old_val == 0:
                        return {'error': 'Cannot calculate growth rate with zero as old value'}
                    growth = ((new_val - old_val) / old_val) * 100
                    return {
                        'operation': 'growth_rate',
                        'old_value': old_val,
                        'new_value': new_val,
                        'growth_percentage': growth,
                        'result': growth
                    }
                
                elif operation == 'percentage':
                    if len(numbers) != 2:
                        return {'error': 'Percentage calculation requires exactly 2 numbers (part, whole)'}
                    part, whole = numbers
                    if whole == 0:
                        return {'error': 'Cannot calculate percentage with zero as whole'}
                    percentage = (part / whole) * 100
                    return {
                        'operation': 'percentage',
                        'part': part,
                        'whole': whole,
                        'percentage': percentage,
                        'result': percentage
                    }
                
                elif operation == 'ratio':
                    if len(numbers) != 2:
                        return {'error': 'Ratio calculation requires exactly 2 numbers'}
                    num1, num2 = numbers
                    if num2 == 0:
                        return {'error': 'Cannot calculate ratio with zero as denominator'}
                    ratio = num1 / num2
                    return {
                        'operation': 'ratio',
                        'numerator': num1,
                        'denominator': num2,
                        'ratio': ratio,
                        'result': ratio
                    }
                
                elif operation == 'sum':
                    result = sum(numbers)
                    return {
                        'operation': 'sum',
                        'numbers': numbers,
                        'result': result
                    }
                
                elif operation == 'average':
                    result = sum(numbers) / len(numbers)
                    return {
                        'operation': 'average',
                        'numbers': numbers,
                        'result': result
                    }
                
                else:
                    return {'error': f'Unknown operation: {operation}'}
                    
            except Exception as e:
                logger.error(f"Calculator error: {e}")
                return {'error': str(e)}
        
        tools.append(AgentTool(
            name="calculator",
            description="Perform financial calculations (growth rates, percentages, ratios, etc.)",
            function=calculator
        ))
        
        # 3. Summarizer Tool
        @tool
        def summarizer(text: str, max_length: int = 300) -> Dict[str, Any]:
            """
            Summarize text content
            
            Args:
                text: Text to summarize
                max_length: Maximum length of summary
                
            Returns:
                Summarized text
            """
            try:
                if not text or len(text) < 50:
                    return {'error': 'Text too short to summarize'}
                
                # Use LLM for summarization
                prompt = f"""
                Summarize the following financial text in {max_length} characters or less:

                {text}

                Focus on key financial metrics, trends, and important information.
                """
                
                summary = self.llm.invoke(prompt)
                
                return {
                    'original_length': len(text),
                    'summary_length': len(summary),
                    'summary': summary,
                    'max_length': max_length
                }
                
            except Exception as e:
                logger.error(f"Summarizer error: {e}")
                return {'error': str(e)}
        
        tools.append(AgentTool(
            name="summarizer",
            description="Summarize financial text content",
            function=summarizer,
            requires_llm=True
        ))
        
        # 4. Comparison Tool
        @tool
        def comparison_tool(data1: Dict[str, Any], data2: Dict[str, Any], metrics: List[str]) -> Dict[str, Any]:
            """
            Compare two sets of financial data
            
            Args:
                data1: First dataset
                data2: Second dataset
                metrics: List of metrics to compare
                
            Returns:
                Comparison results
            """
            try:
                comparison = {
                    'metrics': metrics,
                    'differences': {},
                    'similarities': {},
                    'summary': ''
                }
                
                for metric in metrics:
                    val1 = data1.get(metric)
                    val2 = data2.get(metric)
                    
                    if val1 is not None and val2 is not None:
                        try:
                            diff = float(val2) - float(val1)
                            pct_change = (diff / float(val1)) * 100 if float(val1) != 0 else 0
                            
                            comparison['differences'][metric] = {
                                'value1': val1,
                                'value2': val2,
                                'difference': diff,
                                'percentage_change': pct_change
                            }
                        except (ValueError, TypeError):
                            comparison['similarities'][metric] = {
                                'value1': val1,
                                'value2': val2,
                                'match': val1 == val2
                            }
                
                # Generate summary
                summary_parts = []
                for metric, diff_data in comparison['differences'].items():
                    summary_parts.append(
                        f"{metric}: {diff_data['percentage_change']:.1f}% change"
                    )
                
                comparison['summary'] = " | ".join(summary_parts[:3])  # Top 3 changes
                
                return comparison
                
            except Exception as e:
                logger.error(f"Comparison tool error: {e}")
                return {'error': str(e)}
        
        tools.append(AgentTool(
            name="comparison_tool",
            description="Compare two sets of financial data",
            function=comparison_tool
        ))
        
        return tools
    
    def _build_agent_graph(self) -> StateGraph:
        """Build the LangGraph agent workflow"""
        
        # Create state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self._plan_workflow)
        workflow.add_node("tool_node", ToolNode([tool.function for tool in self.tools]))
        workflow.add_node("analyzer", self._analyze_results)
        workflow.add_node("reporter", self._generate_report)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Define edges
        workflow.add_conditional_edges(
            "planner",
            self._should_use_tools,
            {
                "continue": "tool_node",
                "end": "reporter"
            }
        )
        
        workflow.add_conditional_edges(
            "tool_node",
            self._should_analyze,
            {
                "analyze": "analyzer",
                "continue": "tool_node",
                "end": "reporter"
            }
        )
        
        workflow.add_conditional_edges(
            "analyzer",
            self._should_continue,
            {
                "continue": "tool_node",
                "end": "reporter"
            }
        )
        
        workflow.add_edge("reporter", END)
        
        return workflow.compile(
            checkpointer=None,  # Disable checkpointing to prevent recursion issues
            interrupt_before=[],  # Don't interrupt before any nodes
            interrupt_after=[]    # Don't interrupt after any nodes
        )
    
    def _plan_workflow(self, state: AgentState) -> AgentState:
        """Plan the agent's workflow based on the user's request"""
        messages = state['messages']
        user_query = messages[-1].content
        
        planning_prompt = f"""
        You are a financial analysis agent. Plan a multi-step workflow to answer the following user request:

        User Request: "{user_query}"

        Available Tools:
        {self._format_tools_description()}

        Plan the workflow by identifying:
        1. What information needs to be gathered
        2. What calculations or comparisons are needed
        3. What tools should be used and in what order
        4. What the final output should look like

        Return your plan as a structured response with tool calls if needed.
        """
        
        try:
            plan = self.llm.invoke(planning_prompt)
            
            # Parse plan for tool calls
            tool_calls = self._extract_tool_calls_from_plan(plan)
            
            return {
                **state,
                'current_step': 'planning',
                'tool_calls': tool_calls,
                'messages': messages + [AIMessage(content=plan)]
            }
            
        except Exception as e:
            logger.error(f"Planning error: {e}")
            return {
                **state,
                'error': str(e),
                'current_step': 'error'
            }
    
    def _should_use_tools(self, state: AgentState) -> str:
        """Determine if tools should be used"""
        if state.get('error'):
            return "end"
        
        tool_calls = state.get('tool_calls', [])
        if tool_calls:
            return "continue"
        
        return "end"
    
    def _should_analyze(self, state: AgentState) -> str:
        """Determine if results should be analyzed"""
        # Check if all tool calls are completed
        tool_calls = state.get('tool_calls', [])
        if tool_calls:
            return "continue"
        
        # Check if we have enough information for analysis
        step_results = state.get('step_results', {})
        if len(step_results) > 0:
            return "analyze"
        
        return "end"
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if agent should continue or end"""
        # Check if we have a final answer
        if state.get('final_answer'):
            return "end"
        
        # Check for errors
        if state.get('error'):
            return "end"
        
        return "continue"
    
    def _analyze_results(self, state: AgentState) -> AgentState:
        """Analyze the results from tool calls"""
        step_results = state.get('step_results', {})
        
        analysis_prompt = f"""
        Analyze the following results from tool calls and determine what insights can be drawn:

        Results: {step_results}

        User Request: {state['messages'][-1].content}

        Provide analysis that answers the user's request or identifies what additional information is needed.
        """
        
        try:
            analysis = self.llm.invoke(analysis_prompt)
            
            return {
                **state,
                'current_step': 'analysis',
                'messages': state['messages'] + [AIMessage(content=analysis)]
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                **state,
                'error': str(e)
            }
    
    def _generate_report(self, state: AgentState) -> AgentState:
        """Generate the final report"""
        messages = state['messages']
        step_results = state.get('step_results', {})
        
        report_prompt = f"""
        Generate a comprehensive financial analysis report based on the following information:

        User Request: {messages[-1].content}
        Analysis Results: {step_results}
        Conversation History: {[str(m) for m in messages]}

        Format the report with:
        1. Executive Summary
        2. Key Findings
        3. Analysis and Insights
        4. Conclusions and Recommendations (if applicable)

        Make the report professional and suitable for financial analysis.
        """
        
        try:
            report = self.llm.invoke(report_prompt)
            
            return {
                **state,
                'final_answer': report,
                'current_step': 'reporting'
            }
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {
                **state,
                'error': str(e)
            }
    
    def _format_tools_description(self) -> str:
        """Format tools description for LLM"""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)
    
    def _extract_tool_calls_from_plan(self, plan: str) -> List[Dict[str, Any]]:
        """Extract tool calls from planning response"""
        # Simple parsing - in production, use more robust parsing
        tool_calls = []
        
        # Look for tool call patterns
        import re
        patterns = [
            r'financial_search\((.*?)\)',
            r'calculator\((.*?)\)',
            r'summarizer\((.*?)\)',
            r'comparison_tool\((.*?)\)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, plan, re.IGNORECASE)
            for match in matches:
                tool_name = pattern.split('(')[0]
                tool_calls.append({
                    'name': tool_name,
                    'arguments': match
                })
        
        return tool_calls
    
    async def run_analysis(self, user_query: str, collections: List[str]) -> Dict[str, Any]:
        """
        Run multi-step financial analysis
        
        Args:
            user_query: User's analysis request
            collections: Collections to search
            
        Returns:
            Analysis results
        """
        try:
            # Initialize state
            initial_state = {
                'messages': [HumanMessage(content=user_query)],
                'current_step': 'initial',
                'step_results': {},
                'tool_calls': [],
                'final_answer': None,
                'error': None
            }
            
            # Run the agent with increased recursion limit
            result = await self.graph.ainvoke(
                initial_state,
                config={
                    "recursion_limit": 50,  # Increase recursion limit significantly
                    "max_iterations": 20,   # Add max iterations limit
                    "timeout": 120          # 2 minute timeout
                }
            )
            
            return {
                'success': True,
                'final_answer': result.get('final_answer'),
                'step_results': result.get('step_results', {}),
                'messages': result.get('messages', []),
                'error': result.get('error')
            }
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'final_answer': None,
                'step_results': {},
                'messages': []
            }


def create_agent_interface(agent: FinancialAgent, collections: List[str]):
    """
    Create Streamlit interface for the financial agent
    
    Args:
        agent: FinancialAgent instance
        collections: Available collections for search
    """
    st.subheader("🤖 Multi-Step Agent Analysis")
    
    st.write("""
    **What is this?**
    The AI agent performs complex multi-step financial analysis by breaking down your request into multiple analytical steps. 
    It can plan workflows, search documents, perform calculations, and generate comprehensive reports automatically.
    
    **How it works:**
    1. **Planning**: The agent analyzes your request and creates a step-by-step plan
    2. **Information Gathering**: It searches through your documents for relevant data
    3. **Analysis**: Performs calculations, comparisons, and data processing
    4. **Reporting**: Generates a comprehensive analysis report
    
    **Available Tools:**
    - **Financial Search**: Search across multiple documents simultaneously
    - **Calculator**: Perform financial calculations (growth rates, percentages, ratios)
    - **Summarizer**: Condense large amounts of text into key insights
    - **Comparison Tool**: Compare financial data between different reports
    
    **Example queries:**
    - "Compare Tesla's and Apple's revenue growth over the past year"
    - "Analyze the risk factors affecting both companies"
    - "Calculate and compare profit margins between the reports"
    - "What are the key differences in their cash flow statements?"
    """)
    
    # User input
    user_query = st.text_area(
        "Enter your multi-step analysis request:",
        height=150,
        placeholder="e.g., Analyze the financial performance differences between these companies..."
    )
    
    # Collection selection
    selected_collections = st.multiselect(
        "Select documents for analysis:",
        options=collections,
        default=collections[:2] if len(collections) >= 2 else collections
    )
    
    # Analysis parameters
    col1, col2 = st.columns([2, 1])
    with col1:
        max_steps = st.slider("Maximum analysis steps", 3, 8, 4)
    with col2:
        include_calculations = st.checkbox("Include financial calculations", value=True)
    
    # Run analysis button
    if st.button("🚀 Start Multi-Step Analysis", type="primary"):
        if not user_query.strip():
            st.error("Please enter an analysis request")
            return
        
        if not selected_collections:
            st.error("Please select at least one document")
            return
        
        with st.spinner("🤖 Agent is analyzing your request... This may take 30 seconds to 2 minutes..."):
            try:
                # Run agent analysis
                result = asyncio.run(
                    agent.run_analysis(user_query, selected_collections)
                )
                
                if result['success']:
                    st.success("✅ Analysis completed successfully!")
                    
                    # Display results
                    if result['final_answer']:
                        st.subheader("📊 Final Analysis Report")
                        st.write(result['final_answer'])
                    
                    # Display step-by-step results
                    if result['step_results']:
                        st.subheader("🔍 Step-by-Step Results")
                        for step, data in result['step_results'].items():
                            with st.expander(f"Step: {step}"):
                                st.json(data)
                    
                    # Display conversation history
                    if result['messages']:
                        st.subheader("💬 Agent Conversation")
                        for i, msg in enumerate(result['messages']):
                            role = "User" if isinstance(msg, HumanMessage) else "Agent"
                            with st.expander(f"{role} Message {i+1}"):
                                st.write(msg.content)
                
                else:
                    st.error(f"❌ Analysis failed: {result['error']}")
                    
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
    
    # Agent capabilities info
    with st.expander("ℹ️ Agent Capabilities"):
        st.write("**Available Tools:**")
        for tool in agent.tools:
            st.write(f"- **{tool.name}**: {tool.description}")
        
        st.write("**Analysis Types:**")
        st.write("- Multi-document comparison")
        st.write("- Financial metric calculation")
        st.write("- Trend analysis")
        st.write("- Risk factor identification")
        st.write("- Comprehensive reporting")
