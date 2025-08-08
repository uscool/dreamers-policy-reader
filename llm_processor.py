from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
import re
import json
from typing import List, Dict

class LLMProcessor:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            temperature=0.1,
            max_output_tokens=150
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Based on the following context from policy documents, provide a concise one-line response to the query.
            If the information is not available in the context, state "Information not found in available documents."
            
            Query: {query}
            Context: {context}
            
            Response:"""
        )
    
    def parse_query(self, query: str) -> Dict:
        details = {
            "age": None,
            "gender": None,
            "procedure": None,
            "location": None,
            "policy_duration": None,
            "original_query": query
        }
        
        try:
            if "M" in query or "male" in query.lower():
                details["gender"] = "male"
            elif "F" in query or "female" in query.lower():
                details["gender"] = "female"
            
            age_match = re.search(r'(\d+)[-]?[MF]?', query)
            if age_match:
                details["age"] = int(age_match.group(1))
            
            procedures = ["surgery", "operation", "treatment", "procedure"]
            for proc in procedures:
                if proc in query.lower():
                    words = query.split()
                    for i, word in enumerate(words):
                        if proc in word.lower():
                            details["procedure"] = words[i-1] + " " + word if i > 0 else word
                            break
                    break
            
            locations = ["Pune", "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"]
            for loc in locations:
                if loc.lower() in query.lower():
                    details["location"] = loc
                    break
            
            duration_match = re.search(r'(\d+)[-]?(month|year|yr)', query.lower())
            if duration_match:
                details["policy_duration"] = f"{duration_match.group(1)} {duration_match.group(2)}"
            
            return details
        except Exception as e:
            return {"original_query": query, "error": str(e)}
    
    def process_query(self, query: str, search_results: List[Dict]) -> str:
        try:
            if not search_results:
                return '{"decision": "rejected", "amount": 0, "justification": "No relevant information found in available documents.", "clauses_referenced": []}'
            
            query_details = self.parse_query(query)
            
            context_parts = []
            clauses_referenced = []
            
            for i, result in enumerate(search_results[:3]):
                content = result.get('content', '')
                if content:
                    limited_content = content[:400]
                    context_parts.append(limited_content)
                    clauses_referenced.append(f"Clause {i+1}: {limited_content[:150]}...")
            
            context = "\n\n".join(context_parts)
            
            structured_prompt = f"""
            Based on the following context from policy documents, analyze the query and provide a structured JSON response.
            
            Query Details: {query_details}
            Context: {context}
            
            Return ONLY a valid JSON object with the following structure:
            {{
                "decision": "approved" or "rejected" (use "pending" ONLY if absolutely necessary and you cannot decide, or if topic is sensitive and may be a legal issue for company),
                "amount": number (0 if rejected),
                "justification": "brief explanation",
                "clauses_referenced": ["list of specific clauses used"]
            }}
            
            Important: Return ONLY the JSON object, no additional text.
            JSON Response:"""
            
            response = self.llm.invoke(structured_prompt)
            response_text = response.content.strip()
            
            try:
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                parsed_response = json.loads(response_text)
                return json.dumps(parsed_response, indent=2)
            except json.JSONDecodeError:
                fallback = {
                    "decision": "pending",
                    "amount": 0,
                    "justification": "Unable to parse response. Please check document coverage.",
                    "clauses_referenced": clauses_referenced
                }
                return json.dumps(fallback, indent=2)
            
        except Exception as e:
            error_response = {
                "decision": "error",
                "amount": 0,
                "justification": f"Error processing query: {str(e)}",
                "clauses_referenced": []
            }
            return json.dumps(error_response, indent=2)
    
    def is_available(self) -> bool:
        return bool(os.getenv('GOOGLE_API_KEY')) 