from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os

class LLMProcessor:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # ULTRA FAST: Gemini 2.0 is faster
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            temperature=0.0,  # Deterministic for speed
            max_output_tokens=200,  # Balanced length for detailed but concise responses
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""Answer this question based on the policy context. Provide specific details and relevant information from the document.

Guidelines:
- Be specific and accurate
- Include key numbers, dates, and conditions when mentioned
- Provide 2-3 sentences of relevant detail
- If information is not in the context, say "Information not found in available documents"

Question: {query}
Context: {context}
Answer:"""
        )
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
Answer the following question using only the provided policy context. Keep the answer brief but detailed, accurate, and specific. Do not use formatting, do not reference yourself, and avoid unnecessary words. Always include key numbers, dates, and conditions if present. If the information is not found in the context, respond with: Information not found in available documents.

Question: {query}
Context: {context}
Answer:"""
        )
    
    def is_available(self) -> bool:
        return bool(os.getenv('GOOGLE_API_KEY')) 