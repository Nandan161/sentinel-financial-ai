import streamlit as st
import os
from pathlib import Path
from src.engine import FinancialRAGEngine
from src.utils.ingestor import FinancialIngestor
from src.utils.vector_store import FinancialVectorStore

# --- 1. Page Configuration & State Initialization ---
st.set_page_config(page_title="Sentinel Financial AI", page_icon="🛡️", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_collections" not in st.session_state:
    st.session_state.active_collections = []

st.title("🛡️ Sentinel-Financial-AI")
st.markdown("### Secure, Privacy-First Multi-Document Analysis")

# --- 2. Initialize AI Engine ---
@st.cache_resource
def load_engine():
    return FinancialRAGEngine()

engine = load_engine()

# --- 3. Sidebar: Document Management ---
with st.sidebar:
    st.title("📂 Document Manager")
    
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")

    # --- SECTION 1: SELECT & ACTIVATE (Top) ---
    existing_files = [f for f in os.listdir("data/raw") if f.endswith(".pdf")]
    
    if existing_files:
        st.subheader("📋 Select Reports to Analyze")
        selected_files = st.multiselect(
            "Choose documents:", 
            options=existing_files,
            default=st.session_state.get("last_selected", [])
        )
        
        if selected_files:
            st.write("---")
            st.caption("📄 Quick Actions (Download)")
            for file_name in selected_files:
                col_name, col_btn = st.columns([3, 1])
                col_name.text(f"• {file_name}")
                file_path = os.path.join("data/raw", file_name)
                with open(file_path, "rb") as f:
                    col_btn.download_button(label="💾", data=f, file_name=file_name, key=f"dl_{file_name}")

            if st.button("🚀 Activate Selection", use_container_width=True):
                # Convert filenames to safe collection names for the vector store
                st.session_state.active_collections = ["".join(filter(str.isalnum, Path(f).stem)) for f in selected_files]
                st.session_state.last_selected = selected_files
                st.rerun()
    else:
        st.info("No documents found. Upload one below.")

    st.divider()

    # --- SECTION 2: UPLOAD & INDEX (Bottom) ---
    st.subheader("📤 Upload New Document")
    uploaded_file = st.file_uploader("Upload 10-K (PDF)", type="pdf")
    
    if uploaded_file:
        file_path = os.path.join("data/raw", uploaded_file.name)
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        if st.button("➕ Process & Index"):
            with st.spinner("Indexing..."):
                safe_name = "".join(filter(str.isalnum, Path(uploaded_file.name).stem))
                store = FinancialVectorStore()
                
                # Logical Check: Skip if already done
                if store.collection_exists(safe_name):
                    st.info(f"⚡ {uploaded_file.name} is already indexed.")
                else:
                    ingestor = FinancialIngestor()
                    chunks = ingestor.ingest_document(uploaded_file.name)
                    store.create_store(chunks, collection_name=safe_name)
                
                # Auto-activate the newly uploaded file
                if safe_name not in st.session_state.active_collections:
                    st.session_state.active_collections.append(safe_name)
                
                st.success(f"✅ {uploaded_file.name} ready!")
                st.rerun()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 4. Main Chat Interface ---
if not st.session_state.active_collections:
    st.info("👈 Please select and 'Activate' documents from the sidebar to begin.")
else:
    # 1. Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                st.markdown("---")
                st.markdown("#### 📚 Reference Sources")
                unique_citations = sorted(list(set(
                    f"• {Path(d['metadata'].get('source', 'Unknown')).name} (Page {d['metadata'].get('page', 0) + 1})"
                    for d in message["sources"]
                )))
                for citation in unique_citations:
                    st.markdown(citation)
                with st.expander("🔍 View Source Evidence"):
                    for i, doc_dict in enumerate(message["sources"]):
                        meta = doc_dict.get("metadata", {})
                        fname = Path(meta.get("source", "Unknown")).name
                        page = meta.get("page", 0) + 1
                        st.markdown(f"**Source {i+1}** | *{fname} (Page {page})*")
                        st.info(doc_dict["content"])

    # 2. Handling New Input
    placeholder = "Compare these reports..." if len(st.session_state.active_collections) > 1 else "Ask a question..."
    if prompt := st.chat_input(placeholder):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                result = engine.query(prompt, collection_names=st.session_state.active_collections)
                answer, raw_sources = result["answer"], result["sources"]
                st.markdown(answer)
                
                # Your specific logic for serialization and redaction checking
                serializable_sources = [{"content": d.page_content, "metadata": d.metadata} for d in raw_sources]

                st.markdown("---")
                st.markdown("#### 📚 Reference Sources")
                unique_citations = sorted(list(set(
                    f"• {Path(d['metadata'].get('source', 'Unknown')).name} (Page {d['metadata'].get('page', 0) + 1})"
                    for d in serializable_sources
                )))
                for citation in unique_citations:
                    st.markdown(citation)

                with st.expander("🔍 View Source Evidence (Audit Log)"):
                    for i, doc_dict in enumerate(serializable_sources):
                        meta = doc_dict.get("metadata", {})
                        fname = Path(meta.get("source", "Unknown")).name
                        page = meta.get("page", 0) + 1
                        st.markdown(f"**Source {i+1}** | *{fname} (Page {page})*")
                        
                        source_text = doc_dict.get("content", "")
                        if "[PERSON" not in source_text and "Elon Musk" in source_text:
                            st.warning("⚠️ Warning: This chunk was NOT redacted.")
                        st.info(source_text)
                        
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": serializable_sources})
        st.rerun()