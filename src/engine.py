from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from src.utils.vector_store import FinancialVectorStore

class FinancialRAGEngine:
    def __init__(self):
        self.vector_store = FinancialVectorStore()
        # We don't need to define self.retriever here anymore 
        # because we define it dynamically inside the query() method now!
        # The Brain: Using Llama 3 for reasoning
        self.llm = OllamaLLM(model="llama3")
        

        # The Instruction: Telling the AI how to behave
        self.template = """
        You are a senior financial analyst at Sentinel AI. 
        Use the following pieces of retrieved context to answer the user's question.
        If the answer is not in the context, say that you don't know. 
        Keep the answer professional and concise.

        Context: {context}
        Question: {question}
        
        Answer:"""
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def query(self, user_question, collection_name="tesla_10k_report"):
        # 1. We now tell the retriever which specific 'drawer' (collection) to open
        retriever = self.vector_store.get_retriever(collection_name=collection_name)
        docs = retriever.invoke(user_question)
        
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 2. Build the final prompt using the retrieved context
        formatted_prompt = self.prompt.format(context=context_text, question=user_question)
        
        # 3. Get the final answer from Llama 3
        print(f"\n--- Sentinel is searching in: {collection_name} ---")
        response = self.llm.invoke(formatted_prompt)
        return response

if __name__ == "__main__":
    engine = FinancialRAGEngine()
    
    while True:
        query = input("\nAsk a question about the 10-K (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        
        answer = engine.query(query)
        print(f"\n[Sentinel]: {answer}")