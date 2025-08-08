import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docling.document_converter import DocumentConverter
import json
import os
import pyarrow as pa
import uuid
from datetime import datetime

class PolicyVectorDB:
    def __init__(self, db_path="./policy_vectordb"):
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
        import os as _os
        model_name = _os.getenv('EMBED_MODEL', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name)
        # Removed RecursiveCharacterTextSplitter; custom chunking will be used
        
    def extract_document_content(self, pdf_path):
        try:
            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            return result.document.export_to_markdown()
        except Exception as e:
            print(f"Error extracting document content: {e}")
            return None
    
    def chunk_text(self, text, max_chunk_size=1200):
        """
        Chunk markdown text by headings, keeping all content under each heading together.
        If a section exceeds max_chunk_size, it is further split by paragraphs.
        """
        import re
        if not text:
            return []

        # Split by markdown headings (e.g., #, ##, ###, etc.)
        heading_regex = re.compile(r'(^#+ .*$)', re.MULTILINE)
        parts = heading_regex.split(text)
        chunks = []
        current_chunk = ''
        for i in range(len(parts)):
            part = parts[i]
            if heading_regex.match(part):
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = part + '\n'
            else:
                current_chunk += part
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Further split any chunk that is too large with paragraph-level overlap
        final_chunks = []
        overlap_paragraph = ''
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                paras = chunk.split('\n\n')
                temp = overlap_paragraph
                for para in paras:
                    candidate_len = len(temp) + (2 if temp else 0) + len(para)
                    if candidate_len > max_chunk_size and temp:
                        final_chunks.append(temp.strip())
                        # set overlap to last paragraph of the previous temp
                        last_para = temp.split('\n\n')[-1]
                        temp = (last_para + '\n\n' + para)
                    else:
                        temp = (temp + ('\n\n' if temp else '')) + para
                if temp.strip():
                    final_chunks.append(temp.strip())
                # Prepare overlap from last paragraph of this large section
                try:
                    overlap_paragraph = paras[-1]
                except Exception:
                    overlap_paragraph = ''
            else:
                final_chunks.append(chunk)
        return final_chunks
    
    def create_embeddings(self, texts):
        if not texts:
            return []
        
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()
    
    def create_table(self, table_name="policy_documents"):
        schema = pa.schema([
            ("id", pa.int64()),
            ("chunk_id", pa.int64()),
            ("content", pa.string()),
            ("embedding", pa.list_(pa.float32(), 384)),
            ("source", pa.string()),
            ("chunk_index", pa.int64()),
            ("user_id", pa.string()),
            ("document_id", pa.string()),
            ("upload_timestamp", pa.timestamp('us')),
            ("access_level", pa.string()),
            ("type", pa.string())
        ])
        
        if table_name not in self.db.table_names():
            self.db.create_table(table_name, schema=schema)
        
        self.table = self.db.open_table(table_name)
        return self.table
    
    def add_document(self, pdf_path, user_id, access_level="private", doc_type="policy", table_name="policy_documents", document_id: str | None = None, source_override: str | None = None):
        try:
            if not os.path.exists(pdf_path):
                print(f"PDF file not found: {pdf_path}")
                return False
            
            content = self.extract_document_content(pdf_path)
            if not content:
                print(f"Failed to extract content from {pdf_path}")
                return False
            
            chunks = self.chunk_text(content)
            if not chunks:
                print("No chunks created from document")
                return False
            
            embeddings = self.create_embeddings(chunks)
            if not embeddings:
                print("Failed to create embeddings")
                return False
            
            document_id = document_id or str(uuid.uuid4())
            upload_timestamp = datetime.now()
            
            data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                data.append({
                    "id": i,
                    "chunk_id": i,
                    "content": chunk,
                    "embedding": embedding,
                    "source": source_override or pdf_path,
                    "chunk_index": i,
                    "user_id": user_id,
                    "document_id": document_id,
                    "upload_timestamp": upload_timestamp,
                    "access_level": access_level,
                    "type": doc_type
                })
            
            table = self.create_table(table_name)
            table.add(data)
            
            print(f"Successfully added {len(chunks)} chunks from {pdf_path}")
            print(f"Document ID: {document_id}")
            return document_id
            
        except Exception as e:
            print(f"Error adding document to vector database: {e}")
            return None
    
    def search_similar(self, query, user_id, doc_type=None, table_name="policy_documents", limit=5):
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            table = self.db.open_table(table_name)
            
            filtered_table = table.search(query_embedding).where(f"user_id = '{user_id}'")
            
            if doc_type:
                results = filtered_table.limit(limit).to_pandas()
                results = results[results['type'] == doc_type]
                return results.to_dict('records')
            
            results = filtered_table.limit(limit).to_pandas()
            return results.to_dict('records')
            
        except Exception as e:
            print(f"Error searching vector database: {e}")
            return []

    def search_similar_in_document(self, query, user_id, document_id, table_name: str = "policy_documents", limit: int = 6):
        """
        Search for chunks similar to query within a specific user's document.
        """
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            table = self.db.open_table(table_name)
            where_expr = f"user_id = '{user_id}' AND document_id = '{document_id}'"
            results = table.search(query_embedding).where(where_expr).limit(limit).to_pandas()
            return results.to_dict('records')
        except Exception as e:
            print(f"Error searching document {document_id}: {e}")
            return []
    
    def get_user_documents(self, user_id, doc_type=None, table_name="policy_documents"):
        try:
            table = self.db.open_table(table_name)
            user_docs = table.to_pandas()
            filtered_docs = user_docs[user_docs['user_id'] == user_id]
            
            if doc_type:
                filtered_docs = filtered_docs[filtered_docs['type'] == doc_type]
            
            return filtered_docs
                
        except Exception as e:
            print(f"Error getting user documents: {e}")
            return pd.DataFrame()
    
    def delete_user_documents(self, user_id, document_id=None, table_name="policy_documents"):
        try:
            table = self.db.open_table(table_name)
            
            if document_id:
                table.delete(f"user_id = '{user_id}' AND document_id = '{document_id}'")
                print(f"Deleted document {document_id} for user {user_id}")
            else:
                table.delete(f"user_id = '{user_id}'")
                print(f"Deleted all documents for user {user_id}")
            
            return True
            
        except Exception as e:
            print(f"Error deleting user documents: {e}")
            return False
    
    def get_document_types(self, user_id, table_name="policy_documents"):
        try:
            docs = self.get_user_documents(user_id, table_name=table_name)
            if not docs.empty:
                return docs['type'].unique().tolist()
            return []
        except Exception as e:
            print(f"Error getting document types: {e}")
            return [] 