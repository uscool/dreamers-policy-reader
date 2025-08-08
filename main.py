import os
import sys
import requests
from dotenv import load_dotenv
from user_auth import UserAuth
from vector_db import PolicyVectorDB
from llm_processor import LLMProcessor
import getpass

load_dotenv()

class PolicyParseApp:
    def __init__(self):
        self.user_auth = UserAuth()
        self.vector_db = PolicyVectorDB()
        self.llm_processor = LLMProcessor()
        self.current_user = None
    
    def login_or_register(self):
        """Handle user login or registration"""
        print("PolicyParse - Document Processing System")
        print("=" * 50)
        
        while True:
            print("\n1. Login")
            print("2. Register")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                if self.login():
                    return True
            elif choice == "2":
                self.register()
            elif choice == "3":
                print("Goodbye!")
                return False
            else:
                print("Invalid choice. Please try again.")
    
    def login(self):
        """User login"""
        print("\n--- Login ---")
        user_id = input("Enter User ID: ").strip()
        password = getpass.getpass("Enter Password: ")
        
        if self.user_auth.authenticate_user(user_id, password):
            self.current_user = user_id
            print(f"Welcome, {user_id}!")
            return True
        else:
            print("Login failed. Please check your credentials.")
            return False
    
    def register(self):
        """User registration"""
        print("\n--- Register ---")
        user_id = input("Enter User ID: ").strip()
        
        if self.user_auth.user_exists(user_id):
            print("User ID already exists. Please choose a different one.")
            return
        
        password = getpass.getpass("Enter Password: ")
        confirm_password = getpass.getpass("Confirm Password: ")
        
        if password != confirm_password:
            print("Passwords do not match.")
            return
        
        if self.user_auth.register_user(user_id, password):
            print("Registration successful! You can now login.")
        else:
            print("Registration failed. Please try again.")
    
    def download_pdf(self, url, filename):
        """Download PDF from URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            print(f"Error downloading PDF: {e}")
            return False
    
    def add_document(self):
        """Add a document to the vector database"""
        print("\n--- Add Document ---")
        
        print("Enter PDF source:")
        print("1. Local file path")
        print("2. URL link")
        
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "1":
            file_path = input("Enter local file path: ").strip()
            if not os.path.exists(file_path):
                print("File not found.")
                return
        elif choice == "2":
            url = input("Enter PDF URL: ").strip()
            filename = f"temp_{self.current_user}_{os.path.basename(url)}"
            if not self.download_pdf(url, filename):
                return
            file_path = filename
        else:
            print("Invalid choice.")
            return
        
        doc_type = input("Enter document type (policy/contract/email/claim): ").strip() or "policy"
        access_level = input("Enter access level (private/public): ").strip() or "private"
        
        print(f"Processing document: {file_path}")
        document_id = self.vector_db.add_document(
            file_path, self.current_user, access_level, doc_type
        )
        
        if document_id:
            self.user_auth.add_user_document(
                self.current_user, document_id, os.path.basename(file_path), doc_type, access_level
            )
            print("Document added successfully!")
            
            if choice == "2":
                os.remove(file_path)
        else:
            print("Failed to add document.")
    
    def list_documents(self):
        """List user's documents"""
        print("\n--- Your Documents ---")
        documents = self.user_auth.get_user_documents(self.current_user)
        
        if not documents:
            print("No documents found.")
            return
        
        print(f"{'ID':<36} {'Filename':<30} {'Type':<10} {'Access':<8} {'Added'}")
        print("-" * 100)
        
        for doc in documents:
            print(f"{doc['document_id']:<36} {doc['filename']:<30} {doc['doc_type']:<10} {doc['access_level']:<8} {doc['added_at'].strftime('%Y-%m-%d')}")
    
    def remove_document(self):
        """Remove a document"""
        print("\n--- Remove Document ---")
        documents = self.user_auth.get_user_documents(self.current_user)
        
        if not documents:
            print("No documents to remove.")
            return
        
        print("Your documents:")
        for i, doc in enumerate(documents, 1):
            print(f"{i}. {doc['filename']} ({doc['doc_type']})")
        
        try:
            choice = int(input("Enter document number to remove: ")) - 1
            if 0 <= choice < len(documents):
                doc = documents[choice]
                confirm = input(f"Are you sure you want to remove '{doc['filename']}'? (y/N): ").strip().lower()
                
                if confirm == 'y':
                    self.vector_db.delete_user_documents(self.current_user, doc['document_id'])
                    self.user_auth.remove_user_document(self.current_user, doc['document_id'])
                    print("Document removed successfully!")
                else:
                    print("Operation cancelled.")
            else:
                print("Invalid document number.")
        except ValueError:
            print("Invalid input.")
    
    def query_documents(self):
        """Query user's documents"""
        print("\n--- Query Documents ---")
        
        documents = self.user_auth.get_user_documents(self.current_user)
        if not documents:
            print("No documents available for querying.")
            return
        
        query = input("Enter your query: ").strip()
        if not query:
            print("Query cannot be empty.")
            return
        
        doc_types = self.vector_db.get_document_types(self.current_user)
        if doc_types:
            print(f"Available document types: {', '.join(doc_types)}")
            doc_type = input("Enter document type to filter (or press Enter for all): ").strip() or None
        else:
            doc_type = None
        
        print(f"\nSearching for: '{query}'")
        if doc_type:
            print(f"Filtering by type: {doc_type}")
        
        results = self.vector_db.search_similar(query, self.current_user, doc_type, limit=5)
        
        if results:
            print(f"\nFound {len(results)} results:")
            print("-" * 80)
            
            for i, result in enumerate(results, 1):
                score = result.get('_distance', 'N/A')
                content = result.get('content', '')[:300]
                doc_type_result = result.get('type', 'N/A')
                
                print(f"Result {i} (Score: {score:.4f}, Type: {doc_type_result}):")
                print(f"Content: {content}...")
                print("-" * 80)
            
            # Generate LLM response if available
            if self.llm_processor.is_available():
                print("\n" + "=" * 80)
                print("AI Response:")
                llm_response = self.llm_processor.process_query(query, results)
                print(llm_response)
                print("=" * 80)
            else:
                print("\nNote: Set GOOGLE_API_KEY environment variable for AI-powered responses.")
        else:
            print("No relevant results found.")
    
    def main_menu(self):
        """Main application menu"""
        while True:
            print(f"\n--- Welcome, {self.current_user} ---")
            print("1. Add Document")
            print("2. List Documents")
            print("3. Remove Document")
            print("4. Query Documents")
            print("5. Logout")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                self.add_document()
            elif choice == "2":
                self.list_documents()
            elif choice == "3":
                self.remove_document()
            elif choice == "4":
                self.query_documents()
            elif choice == "5":
                print("Logging out...")
                self.current_user = None
                break
            else:
                print("Invalid choice. Please try again.")

def main():
    app = PolicyParseApp()
    
    if app.login_or_register():
        app.main_menu()

if __name__ == "__main__":
    main() 