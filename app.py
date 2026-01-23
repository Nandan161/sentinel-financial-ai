import streamlit as st
import os
from src.engine import FinancialRAGEngine
from src.utils.ingestor import FinancialIngestor
from src.utils.vector_store import FinancialVectorStore

# --- 1. Page Configuration & State Initialization ---
st.set_page_config(page_title="Sentinel Financial AI", page_icon="🛡️", layout="wide")

# Initialize chat history early
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize document tracking
if "current_collection" not in st.session_state:
    st.session_state.current_collection = "tesla_10k_report"

st.title("🛡️ Sentinel-Financial-AI")
st.markdown("### Secure, Privacy-First Financial Document Analysis")

# --- 2. Initialize AI Engine ---
@st.cache_resource
def load_engine():
    return FinancialRAGEngine()

engine = load_engine()

# --- 3. Sidebar: Document Management ---
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_file = st.file_uploader("Upload a Financial PDF (10-K, etc.)", type="pdf")
    
    if uploaded_file:
        file_path = os.path.join("data/raw", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Process & Index Document"):
            with st.spinner("Processing..."):
                # Clean name for ChromaDB
                safe_name = "".join(filter(str.isalnum, uploaded_file.name))
                
                ingestor = FinancialIngestor()
                chunks = ingestor.ingest_document(uploaded_file.name)
                
                store = FinancialVectorStore()
                store.create_store(chunks, collection_name=safe_name)
                
                st.session_state.current_collection = safe_name
                st.success(f"✅ {uploaded_file.name} is Ready!")

# --- 4. Chat Interface (History First) ---
# This loop ensures your old prompts and answers stay visible on screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. Handling New User Input ---
if prompt := st.chat_input("Ask about the report..."):
    # 1. Immediately display and save the user prompt
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate and display the assistant response
    current_doc = st.session_state.current_collection
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            response = engine.query(prompt, collection_name=current_doc)
            st.markdown(response)
    
    # 3. Save the response to history
    st.session_state.messages.append({"role": "assistant", "content": response})