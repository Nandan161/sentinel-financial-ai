# app.py - PRODUCTION VERSION

import streamlit as st
import os
import logging
from pathlib import Path
from datetime import datetime
from src.engine import FinancialRAGEngine
from src.utils.ingestor import FinancialIngestor, DocumentProcessingError
from src.utils.vector_store import FinancialVectorStore

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
    initial_sidebar_state="expanded"
)

# --- 2. Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_collections" not in st.session_state:
    st.session_state.active_collections = []
    
if "processing_errors" not in st.session_state:
    st.session_state.processing_errors = []

# --- 3. Helper Functions ---
def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal"""
    # Get basename to prevent directory traversal
    safe_name = os.path.basename(filename)
    # Remove any remaining path separators
    safe_name = safe_name.replace('/', '').replace('\\', '')
    return safe_name

def validate_upload(uploaded_file) -> tuple[bool, str]:
    """
    Validate uploaded file
    
    Returns:
        (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "No file provided"
    
    # Check extension
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f}MB). Maximum: {MAX_UPLOAD_SIZE_MB}MB"
    
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

# --- 5. UI Header ---
st.title("🛡️ Sentinel-Financial-AI")
st.markdown("### Secure, Privacy-First Multi-Document Analysis")

# Show warning if engine failed to load
if engine is None:
    st.error("⚠️ System is not fully operational. Please contact administrator.")
    st.stop()

# --- 6. Sidebar: Document Management ---
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

# --- 7. Main Chat Interface ---
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
        
        # Generate response
        with st.chat_message("assistant"):
            try:
                with st.spinner("🔍 Analyzing documents..."):
                    # Query the engine
                    result = engine.query(
                        prompt, 
                        collection_names=st.session_state.active_collections
                    )
                    
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

# --- 8. Footer ---
st.divider()
st.caption("🛡️ Sentinel Financial AI • Privacy-First Document Analysis • All sensitive data is automatically redacted")