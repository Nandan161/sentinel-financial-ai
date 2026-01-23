from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from src.utils.vector_store import FinancialVectorStore

class FinancialRAGEngine:
    def __init__(self):
        # The Brain: Using Llama 3 for reasoning
        self.llm = OllamaLLM(model="llama3")
        
        # The Memory: Accessing our ChromaDB
        self.vector_store = FinancialVectorStore()
        self.retriever = self.vector_store.get_retriever()

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

    def query(self, user_question):
        # 1. Get relevant chunks from memory
        docs = self.retriever.invoke(user_question)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 2. Build the final prompt
        formatted_prompt = self.prompt.format(context=context_text, question=user_question)
        
        # 3. Get response from Llama 3
        print("\n--- Sentinel is thinking... ---")
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