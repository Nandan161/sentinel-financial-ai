import streamlit as st
import os
import logging
from pathlib import Path
from datetime import datetime
from src.engine import FinancialRAGEngine
from src.utils.ingestor import FinancialIngestor, DocumentProcessingError
from src.utils.vector_store import FinancialVectorStore
from src.integration.advanced_features import AdvancedRAGSystem, create_security_dashboard, create_graph_visualization, create_agent_interface, create_feature_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentinel_financial.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
MAX_UPLOAD_SIZE_MB = 100
ALLOWED_EXTENSIONS = ['.pdf']

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Sentinel Financial AI", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Sentinel Financial AI - Advanced Document Analysis System"
    }
)

# --- 2. Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_collections" not in st.session_state:
    st.session_state.active_collections = []
    
if "processing_errors" not in st.session_state:
    st.session_state.processing_errors = []

if "current_page" not in st.session_state:
    st.session_state.current_page = "main"

# --- 3. Helper Functions ---
def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and injection attacks"""
    if not filename:
        return ""
    
    # Remove path separators to prevent directory traversal
    safe_name = os.path.basename(filename)
    safe_name = safe_name.replace('/', '').replace('\\', '').replace('..', '')
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '&', ';', '$', '`', '\\']
    for char in dangerous_chars:
        safe_name = safe_name.replace(char, '')
    
    # Remove leading dots to prevent hidden files
    safe_name = safe_name.lstrip('.')
    
    # Ensure filename is not empty after sanitization
    if not safe_name:
        return "uploaded_file"
    
    # Limit filename length
    if len(safe_name) > 255:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:250] + ext
    
    return safe_name

def validate_upload(uploaded_file) -> tuple[bool, str]:
    """
    Validate uploaded file with comprehensive security checks
    
    Returns:
        (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "No file provided"
    
    # Check filename
    if not uploaded_file.name:
        return False, "File has no name"
    
    # Sanitize filename
    safe_filename = sanitize_filename(uploaded_file.name)
    if not safe_filename:
        return False, "Invalid filename"
    
    # Check extension
    ext = Path(safe_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size
    try:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_UPLOAD_SIZE_MB:
            return False, f"File too large ({file_size_mb:.1f}MB). Maximum: {MAX_UPLOAD_SIZE_MB}MB"
        
        if file_size_mb == 0:
            return False, "File is empty"
    except Exception as e:
        logger.error(f"Error checking file size: {e}")
        return False, "Unable to read file size"
    
    # Additional security checks
    try:
        # Check if file is actually a PDF (basic magic number check)
        uploaded_file.seek(0)
        header = uploaded_file.read(4)
        uploaded_file.seek(0)  # Reset position
        
        if header != b'%PDF':
            return False, "File is not a valid PDF document"
    except Exception as e:
        logger.error(f"Error validating PDF format: {e}")
        return False, "Unable to validate file format"
    
    return True, ""

def log_query(query: str, collections: list, success: bool):
    """Log queries for audit trail"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'query': query[:100],  # Truncate long queries
        'collections': collections,
        'success': success
    }
    logger.info(f"Query: {log_entry}")

# --- 4. Initialize Engine with Error Handling ---
@st.cache_resource
def load_engine():
    try:
        engine = FinancialRAGEngine()
        logger.info("RAG Engine initialized successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize RAG Engine: {e}", exc_info=True)
        st.error("⚠️ Failed to initialize AI engine. Please check system configuration.")
        return None

engine = load_engine()

# --- 5. Initialize Advanced System ---
@st.cache_resource
def load_advanced_system():
    try:
        advanced_system = AdvancedRAGSystem(engine)
        logger.info("Advanced RAG system loaded successfully")
        return advanced_system
    except Exception as e:
        logger.error(f"Failed to initialize advanced system: {e}")
        return None

advanced_system = load_advanced_system()

# --- 6. UI Header ---
st.title("🛡️ Sentinel-Financial-AI")
st.markdown("### Secure, Privacy-First Multi-Document Analysis")

# Show warning if engine failed to load
if engine is None:
    st.error("⚠️ System is not fully operational. Please contact administrator.")
    st.stop()

# --- 7. Top Navigation Bar ---
st.divider()
st.markdown("### 🧭 Navigation")

# Home button at the top
col_home, col_features = st.columns([1, 3])

with col_home:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = "main"
        st.rerun()

with col_features:
    st.markdown("#### 🚀 Advanced Features")

# Navigation buttons with proper state management
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🛡️ Security Dashboard", use_container_width=True):
        st.session_state.current_page = "security_dashboard"
        st.rerun()

with col2:
    if st.button("🕸️ Knowledge Graph", use_container_width=True):
        st.session_state.current_page = "knowledge_graph"
        st.rerun()

with col3:
    if st.button("🤖 Multi-Step Agent", use_container_width=True):
        st.session_state.current_page = "multi_step_agent"
        st.rerun()

with col4:
    if st.button("📋 Feature Comparison", use_container_width=True):
        st.session_state.current_page = "feature_comparison"
        st.rerun()

st.divider()

# --- 8. Navigation Sidebar ---
with st.sidebar:
    st.title("📂 Document Manager")
    
    # Ensure data directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    # Show system statistics
    with st.expander("📊 System Statistics"):
        try:
            existing_files = [f for f in os.listdir("data/raw") if f.endswith(".pdf")]
            st.metric("Documents Indexed", len(existing_files))
            st.metric("Active Documents", len(st.session_state.active_collections))
            st.metric("Total Queries", len(st.session_state.messages) // 2)
        except Exception as e:
            st.error(f"Error loading statistics: {e}")
    
    st.divider()
    
    # --- SECTION 1: Select & Activate ---
    st.subheader("📋 Select Reports to Analyze")
    
    existing_files = [f for f in os.listdir("data/raw") if f.endswith(".pdf")]
    
    if existing_files:
        selected_files = st.multiselect(
            "Choose documents:", 
            options=existing_files,
            default=st.session_state.get("last_selected", []),
            help="Select one or more documents to query"
        )
        
        if selected_files:
            st.caption("📄 Quick Actions")
            for file_name in selected_files:
                col_name, col_btn = st.columns([3, 1])
                col_name.text(f"• {file_name}")
                
                try:
                    file_path = os.path.join("data/raw", file_name)
                    with open(file_path, "rb") as f:
                        col_btn.download_button(
                            label="💾", 
                            data=f, 
                            file_name=file_name, 
                            key=f"dl_{file_name}",
                            help="Download original file"
                        )
                except Exception as e:
                    logger.error(f"Error creating download button: {e}")

            if st.button("🚀 Activate Selection", use_container_width=True, type="primary"):
                try:
                    # Convert filenames to safe collection names
                    st.session_state.active_collections = [
                        "".join(filter(str.isalnum, Path(f).stem)) 
                        for f in selected_files
                    ]
                    st.session_state.last_selected = selected_files
                    logger.info(f"Activated collections: {st.session_state.active_collections}")
                    st.success(f"✅ Activated {len(selected_files)} document(s)")
                    st.session_state.current_page = "main"
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error activating documents: {e}")
                    st.error(f"Failed to activate documents: {e}")
    else:
        st.info("No documents found. Upload one below.")

    st.divider()

    # --- SECTION 2: Upload & Index ---
    st.subheader("📤 Upload New Document")
    
    uploaded_file = st.file_uploader(
        "Upload Financial Report (PDF)", 
        type=['pdf'],
        help=f"Maximum file size: {MAX_UPLOAD_SIZE_MB}MB"
    )
    
    if uploaded_file:
        # Validate file
        is_valid, error_msg = validate_upload(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
        else:
            # Show file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📄 {uploaded_file.name} ({file_size_mb:.1f}MB)")
            
            # Sanitize filename
            safe_filename = sanitize_filename(uploaded_file.name)
            file_path = os.path.join("data/raw", safe_filename)
            
            # Save file if not exists
            if not os.path.exists(file_path):
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    logger.info(f"Saved uploaded file: {safe_filename}")
                except Exception as e:
                    logger.error(f"Error saving file: {e}")
                    st.error(f"Failed to save file: {e}")
            
            # Process button
            if st.button("➕ Process & Index", type="primary", use_container_width=True):
                try:
                    with st.spinner("🔄 Processing document..."):
                        safe_name = "".join(filter(str.isalnum, Path(safe_filename).stem))
                        store = FinancialVectorStore()
                        
                        # Check if already indexed
                        if store.collection_exists(safe_name):
                            st.info(f"⚡ {safe_filename} is already indexed.")
                            logger.info(f"Document already indexed: {safe_name}")
                        else:
                            # Process document
                            ingestor = FinancialIngestor()
                            
                            progress_bar = st.progress(0, text="Loading PDF...")
                            chunks = ingestor.ingest_document(safe_filename)
                            
                            progress_bar.progress(50, text="Creating embeddings...")
                            store.create_store(chunks, collection_name=safe_name)
                            
                            progress_bar.progress(100, text="Complete!")
                            
                            # Show statistics
                            stats = ingestor.get_statistics()
                            st.success(
                                f"✅ Processed {stats['total_pages']} pages into "
                                f"{stats['total_chunks']} searchable chunks "
                                f"({stats['total_redactions']} sensitive items redacted)"
                            )
                            
                            logger.info(f"Successfully indexed: {safe_name}")
                        
                        # Auto-activate
                        if safe_name not in st.session_state.active_collections:
                            st.session_state.active_collections.append(safe_name)
                        
                        st.session_state.current_page = "main"
                        st.rerun()
                        
                except DocumentProcessingError as e:
                    logger.error(f"Processing error: {e}")
                    st.error(f"❌ Processing failed: {e}")
                    st.session_state.processing_errors.append({
                        'file': safe_filename,
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
                    
                except Exception as e:
                    logger.error(f"Unexpected error: {e}", exc_info=True)
                    st.error(f"❌ Unexpected error: {e}")

    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        logger.info("Chat history cleared")
        st.rerun()
    
    # Show recent errors if any
    if st.session_state.processing_errors:
        with st.expander("⚠️ Recent Errors", expanded=False):
            for error in st.session_state.processing_errors[-5:]:  # Last 5 errors
                st.error(f"{error['file']}: {error['error']}")

# Reset page state if no documents are active
if not st.session_state.active_collections and st.session_state.current_page != "main":
    st.session_state.current_page = "main"
    st.rerun()

# --- 9. Page Routing ---
current_page = st.session_state.current_page

if current_page == "security_dashboard" and advanced_system:
    if advanced_system.features_enabled['evaluation']:
        create_security_dashboard(advanced_system.evaluator)
    else:
        st.error("Security Dashboard feature is disabled")

elif current_page == "knowledge_graph" and advanced_system:
    if advanced_system.features_enabled['graph_rag']:
        # Get available collections
        collections = st.session_state.active_collections
        if collections:
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
        else:
            st.info("Please activate documents first to build a knowledge graph")
    else:
        st.error("Knowledge Graph feature is disabled")

elif current_page == "multi_step_agent" and advanced_system:
    if advanced_system.features_enabled['agent']:
        # Get available collections
        collections = st.session_state.active_collections
        if collections:
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
                
                elif not selected_collections:
                    st.error("Please select at least one document")
                
                else:
                    with st.spinner("🤖 Agent is analyzing your request... This may take 30 seconds to 2 minutes..."):
                        try:
                            # Run agent analysis
                            result = advanced_system.run_agent_analysis(user_query, selected_collections)
                            
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
                            
                            else:
                                st.error(f"❌ Analysis failed: {result['error']}")
                                
                        except Exception as e:
                            st.error(f"❌ Unexpected error: {e}")
            
            # Agent capabilities info
            with st.expander("ℹ️ Agent Capabilities"):
                st.write("**Available Tools:**")
                for tool in advanced_system.agent.tools:
                    st.write(f"- **{tool.name}**: {tool.description}")
                
                st.write("**Analysis Types:**")
                st.write("- Multi-document comparison")
                st.write("- Financial metric calculation")
                st.write("- Trend analysis")
                st.write("- Risk factor identification")
                st.write("- Comprehensive reporting")
        else:
            st.info("Please activate documents first to use the multi-step agent")
    else:
        st.error("Multi-Step Agent feature is disabled")

elif current_page == "feature_comparison":
    create_feature_comparison()

else:
    # --- 10. Main Chat Interface ---
    if not st.session_state.active_collections:
        st.info("👈 Please select and **Activate** documents from the sidebar to begin analysis.")
        
        # Show helpful tips
        with st.expander("💡 Quick Start Guide"):
            st.markdown("""
            **Getting Started:**
            1. Upload a financial report (10-K, 10-Q, etc.) using the sidebar
            2. Click **Process & Index** to prepare the document
            3. Select the document and click **Activate Selection**
            4. Start asking questions about the report!
            
            **Example Questions:**
            - What was the revenue for the most recent quarter?
            - Summarize the risk factors
            - What are the main sources of revenue?
            - Compare Q3 and Q4 performance (with multiple docs)
            """)
    else:
        # Display active documents
        st.info(f"📊 Analyzing: {', '.join(st.session_state.get('last_selected', []))}")
        
        # Add note about automatic evaluation
        st.info("""
        💡 **Pro Tip:** Your queries are automatically evaluated for quality! 
        Check the **🛡️ Security Dashboard** to see performance metrics like 
        Faithfulness, Answer Relevance, and Context Quality.
        """)
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    st.markdown("---")
                    st.markdown("#### 📚 Reference Sources")
                    
                    # Unique citations
                    unique_citations = sorted(list(set(
                        f"• {Path(d['metadata'].get('source', 'Unknown')).name} "
                        f"(Page {d['metadata'].get('page', 0) + 1})"
                        for d in message["sources"]
                    )))
                    
                    for citation in unique_citations:
                        st.markdown(citation)
                    
                    # Expandable source details
                    with st.expander("🔍 View Source Evidence"):
                        for i, doc_dict in enumerate(message["sources"]):
                            meta = doc_dict.get("metadata", {})
                            fname = Path(meta.get("source", "Unknown")).name
                            page = meta.get("page", 0) + 1
                            
                            st.markdown(f"**Source {i+1}** | *{fname} (Page {page})*")
                            st.info(doc_dict["content"])

        # Chat input
        placeholder = (
            "Compare these reports..." 
            if len(st.session_state.active_collections) > 1 
            else "Ask a question about the document..."
        )
        
        if prompt := st.chat_input(placeholder):
            # Display user message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response with advanced system
            with st.chat_message("assistant"):
                try:
                    with st.spinner("🔍 Analyzing documents..."):
                        # Use advanced system for evaluation
                        result = advanced_system.query_with_evaluation(
                            prompt, 
                            st.session_state.active_collections,
                            enable_evaluation=True
                        )
                        
                        # Log the evaluation result for debugging
                        if 'evaluation' in result:
                            logger.info(f"Evaluation result: {result['evaluation']}")
                        
                        answer = result["answer"]
                        raw_sources = result["sources"]
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Log query
                        log_query(prompt, st.session_state.active_collections, True)
                        
                        # Serialize sources
                        serializable_sources = [
                            {
                                "content": d.page_content, 
                                "metadata": d.metadata
                            } 
                            for d in raw_sources
                        ]
                        
                        # Display sources
                        if serializable_sources:
                            st.markdown("---")
                            st.markdown("#### 📚 Reference Sources")
                            
                            unique_citations = sorted(list(set(
                                f"• {Path(d['metadata'].get('source', 'Unknown')).name} "
                                f"(Page {d['metadata'].get('page', 0) + 1})"
                                for d in serializable_sources
                            )))
                            
                            for citation in unique_citations:
                                st.markdown(citation)
                            
                            # Source evidence with redaction verification
                            with st.expander("🔍 View Source Evidence (Audit Log)"):
                                for i, doc_dict in enumerate(serializable_sources):
                                    meta = doc_dict.get("metadata", {})
                                    fname = Path(meta.get("source", "Unknown")).name
                                    page = meta.get("page", 0) + 1
                                    
                                    st.markdown(f"**Source {i+1}** | *{fname} (Page {page})*")
                                    
                                    # Redaction verification
                                    source_text = doc_dict.get("content", "")
                                    if "[PERSON" not in source_text and any(
                                        name in source_text 
                                        for name in ["Elon Musk", "Jeff Bezos", "Tim Cook"]
                                    ):
                                        st.warning("⚠️ Warning: Potential unredacted content detected")
                                        logger.warning(f"Unredacted content in {fname} page {page}")
                                    
                                    st.info(source_text)
                        
                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer, 
                            "sources": serializable_sources
                        })
                        
                except Exception as e:
                    logger.error(f"Query failed: {e}", exc_info=True)
                    st.error(f"❌ Failed to process query: {e}")
                    log_query(prompt, st.session_state.active_collections, False)
            
            st.rerun()

# --- 11. Footer ---
st.divider()
st.caption("🛡️ Sentinel Financial AI • Privacy-First Document Analysis • All sensitive data is automatically redacted")
