from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.vector_store import FinancialVectorStore
from pathlib import Path

class FinancialRAGEngine:
    def __init__(self):
        self.vector_store = FinancialVectorStore()
        self.llm = OllamaLLM(model="llama3", temperature=0)

        # UPDATED TEMPLATE: Dynamically handles multiple documents
        self.template = """### System:
You are a Financial Analyst. You have been given chunks from multiple reports.
Your task is to compare metrics (like Revenue, Profit, etc.) between these reports.

RULES:
1. Identify companies based on the 'DOCUMENT SOURCE' labels provided in the context.
2. If the data is redacted (e.g., [PERSON_NAME]), leave it redacted.
3. If the reports are for different periods, note that in your comparison.
4. If you see data for Company A and Company B, you ARE allowed to compare them.

### Context:
{context}

### Question:
{question}

### Answer:"""
        
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def query(self, user_question, collection_names):
        if isinstance(collection_names, str):
            collection_names = [collection_names]

        all_docs = []
        # We track which files were actually retrieved to help the AI map them
        found_sources = set()

        for coll in collection_names:
            try:
                retriever = self.vector_store.get_retriever(collection_name=coll)
                chunks = retriever.invoke(user_question)
                print(f"DEBUG: Found {len(chunks)} chunks for collection {coll}") # Check your terminal!
                
                for chunk in chunks:
                    # Capture the filename for the dynamic prompt
                    fname = Path(chunk.metadata.get("source", coll)).name
                    found_sources.add(fname)
                    
                    # INJECT DYNAMIC LABEL: This tells the AI exactly which company this chunk belongs to
                    chunk.page_content = f">>> DATA FROM REPORT: {fname} <<<\n{chunk.page_content}"
                
                all_docs.extend(chunks)
            except Exception as e:
                print(f"Error: {e}")

        # Combine Context
        context_text = "\n\n".join([doc.page_content for doc in all_docs])
        
        # FINAL SAFEGUARD: Inform the AI which files are available
        if found_sources:
            header = f"The following reports are currently available for analysis: {', '.join(found_sources)}\n\n"
            context_text = header + context_text

        if not all_docs:
            return {
                "answer": "I found no data matching your query in the selected reports. Try asking about specific metrics like 'revenue' or 'expenses'.",
                "sources": []
            }

        chain = self.prompt | self.llm
        response = chain.invoke({"context": context_text, "question": user_question})
        
        return {"answer": response, "sources": all_docs}