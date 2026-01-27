# 🛡️ Sentinel-Financial-AI
> **Enterprise-Grade Multi-Agent RAG for Secure Financial Analysis**

Sentinel-Financial-AI is a privacy-first Retrieval-Augmented Generation (RAG) system designed to analyze sensitive financial documents (like SEC 10-K filings) without leaking PII (Personally Identifiable Information). By processing all data locally, it ensures 100% data sovereignty for high-stakes financial environments.

---

## 🏗️ System Architecture
The project is built on a "Privacy-First" pipeline:

1. **Ingestion Layer:** Extracts text from complex PDFs and handles semantic chunking.
2. **Security Layer:** Scans for PII using Microsoft Presidio and spaCy to redact names, emails, and locations.
3. **Vector Intelligence:** Generates high-dimensional embeddings using `nomic-embed-text`.
4. **Multi-Collection Storage:** Persists vectors in isolated ChromaDB collections to prevent cross-document data leakage.
5. **Contextual Inference:** Orchestrates local LLMs (Llama 3) via Ollama for private, source-grounded analysis.



---

## 🛠️ Technical Stack
* **Language:** Python 3.10+
* **AI Framework:** LangChain
* **LLM / Embeddings:** Ollama (Llama 3, Nomic-Embed-Text)
* **Vector DB:** ChromaDB
* **Security:** Microsoft Presidio & spaCy
* **UI:** Streamlit

---

## 🚀 Getting Started

### Prerequisites
* [Ollama](https://ollama.ai/) installed and running locally.
* Python 3.10 or higher.

### Installation
1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/yourusername/sentinel-financial-ai.git](https://github.com/yourusername/sentinel-financial-ai.git)
   cd sentinel-financial-ai