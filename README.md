# 🛡️ Sentinel-Financial-AI
> **Enterprise-Grade Multi-Agent RAG for Secure Financial Analysis**

Sentinel is a privacy-first Retrieval-Augmented Generation (RAG) system designed to analyze sensitive financial documents (like SEC 10-K filings) without leaking PII (Personally Identifiable Information).

---

## 🏗️ System Architecture
The project is built on a "Privacy-First" pipeline:
1. **Ingestion Layer:** Extracts text from complex PDFs.
2. **Security Layer:** Scans for PII using Microsoft Presidio and redacts names, emails, and locations.
3. **Vector Intelligence:** Chunks data and generates semantic embeddings using `nomic-embed-text`.
4. **Storage:** Persists high-dimensional vectors in a local ChromaDB instance.
5. **Inference:** Orchestrates local LLMs (Llama 3) via Ollama for 100% data sovereignty.



---

## 🛠️ Technical Stack
- **Language:** Python 3.10+
- **AI Framework:** LangChain
- **LLM / Embeddings:** Ollama (Llama 3, Nomic-Embed-Text)
- **Vector DB:** ChromaDB
- **Security:** Microsoft Presidio & Spacy
- **Data Parsing:** PyPDF

---

## 📈 Development Journey (DevLog)

### ✅ Milestone 1: The Privacy Shield (Completed)
- **The Challenge:** Financial documents contain sensitive data that shouldn't be sent to cloud LLMs.
- **The Solution:** Implemented a `FinancialRedactor` class.
- **Result:** Successfully automated the masking of names and locations using NLP.

### ✅ Milestone 2: Data Pipeline & Vectorization (Completed)
- **The Challenge:** Parsing 100+ page PDFs into a format an AI can "search."
- **The Solution:** Built a `FinancialIngestor` that uses Recursive Character Splitting (1000 char chunks).
- **Result:** Processed Tesla’s 2024 10-K into **477 secure, searchable chunks** stored in ChromaDB.

### ✅ Milestone 3: The RAG Intelligence Engine (Completed)
- **The Challenge:** Ensuring the LLM stays "grounded" to the financial data and doesn't hallucinate.
- **The Solution:** Implemented a Retrieval-Augmented Generation (RAG) loop using `langchain-core` and Llama 3.
- **Testing Result:** - **In-Scope:** Successfully identified technical risks (4680 battery cells).
  - **Out-of-Scope:** Correctly refused to answer non-financial questions (e.g., baking recipes).
  - **Privacy:** Maintained integrity by not attempting to "guess" redacted PII.

  ## 🚀 Recent Updates
- **Multi-Document Management:** Added support for independent vector collections. Users can now index and switch between different financial reports (e.g., Apple vs. Tesla) without data leakage.
- **Dynamic Retrieval:** Refactored the engine to instantiate retrievers on-the-fly based on the active document.
- **Enhanced Chat UI:** Implemented `st.session_state` to maintain a persistent chat history, providing a seamless user experience.

## 🛠️ Technical Stack
- **LLM:** Llama 3 (via Ollama)
- **Vector DB:** ChromaDB
- **Framework:** LangChain & Streamlit
- **Privacy:** In-built PII Redaction layer