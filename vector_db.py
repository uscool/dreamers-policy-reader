import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import os
import pyarrow as pa
import uuid
from datetime import datetime
from typing import List
import torch

class PolicyVectorDB:
    def __init__(self, db_path="./policy_vectordb"):
        self.db_path = db_path
        self.db = lancedb.connect(db_path)

        model_name = os.getenv('EMBED_MODEL', 'all-MiniLM-L6-v2')
        
        device = 'cpu'
        print(f"Using device: {device} for embeddings (ULTRA FAST)")
        
        self.embedding_model = SentenceTransformer(
            model_name, 
            device=device,
            cache_folder=None,  # Disable caching
        )

        self._init_document_converters()
        self._heading_regex = None
        self._max_chunk_size_default = 5000

    def _init_document_converters(self):
        # PDF converters
        try:
            import PyPDF2
            self._pypdf2_available = True
        except ImportError:
            self._pypdf2_available = False
            
        try:
            import pdfplumber
            self._pdfplumber_available = True
        except ImportError:
            self._pdfplumber_available = False
        
        # DOCX converter
        try:
            import docx
            self._docx_available = True
        except ImportError:
            self._docx_available = False
        
        # TXT converter (always available)
        self._txt_available = True

    def get_embedding_dim(self) -> int:
        try:
            return int(getattr(self.embedding_model, 'get_sentence_embedding_dimension', lambda: 0)())
        except Exception:
            return 384

    def get_default_table_name(self) -> str:
        return f"policy_documents_{self.get_embedding_dim()}"
        
    def extract_document_content(self, file_path: str) -> str | None:
        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._extract_pdf_content(file_path)
        elif file_ext == '.docx':
            return self._extract_docx_content(file_path)
        elif file_ext == '.txt':
            return self._extract_txt_content(file_path)
        else:
            return self._extract_txt_content(file_path)
    
    def _extract_pdf_content(self, pdf_path: str) -> str | None:
        """Extract text from PDF using only the fastest methods."""
        methods = [
            self._extract_with_pdfplumber,
            self._extract_with_pypdf2,
        ]
        for method in methods:
            try:
                result = method(pdf_path)
                if result and len(result.strip()) > 100:
                    return result
            except Exception as e:
                print(f"PDF method {method.__name__} failed: {e}")
                continue
        
        print("All PDF extraction methods failed")
        return None
    
    def _extract_docx_content(self, docx_path: str) -> str | None:
        if not self._docx_available:
            return None
            
        try:
            import docx
            doc = docx.Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"DOCX extraction failed: {e}")
            return None
    
    def _extract_txt_content(self, txt_path: str) -> str | None:
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                # Try with different encoding
                with open(txt_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                print(f"TXT extraction failed: {e}")
                return None
        except Exception as e:
            print(f"TXT extraction failed: {e}")
            return None
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str | None:
        if not self._pdfplumber_available:
            return None
        import pdfplumber
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
            return None
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str | None:
        if not self._pypdf2_available:
            return None
        import PyPDF2
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
            return None
    
    def _extract_with_docling(self, pdf_path: str) -> str | None:
        if not hasattr(self, '_docling_converter') or not self._docling_converter:
            return None
        try:
            result = self._docling_converter.convert(pdf_path)
            return result.document.export_to_markdown()
        except Exception as e:
            print(f"docling extraction failed: {e}")
            return None
    
    def _get_heading_regex(self):
        if self._heading_regex is None:
            import re as _re
            self._heading_regex = _re.compile(r'(^#+ .*$)', _re.MULTILINE)
        return self._heading_regex

    def chunk_text(self, text: str, max_chunk_size: int | None = None) -> List[str]:
        import re
        if not text:
            return []

        effective_max = max_chunk_size or self._max_chunk_size_default
        heading_regex = self._get_heading_regex()
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

        final_chunks = []
        for chunk in chunks:
            if len(chunk) > effective_max:
                paras = chunk.split('\n\n')
                temp = ''
                for para in paras:
                    if len(temp) + len(para) > effective_max and temp:
                        final_chunks.append(temp.strip())
                        temp = para
                    else:
                        temp = (temp + '\n\n' + para) if temp else para
                if temp.strip():
                    final_chunks.append(temp.strip())
            else:
                final_chunks.append(chunk)

        if len(final_chunks) == 1 and len(final_chunks[0]) > effective_max:
            single_chunk = final_chunks[0]
            final_chunks = []
            sentences = re.split(r'[.!?]+', single_chunk)
            temp = ''
            for sentence in sentences:
                if len(temp) + len(sentence) > effective_max and temp:
                    final_chunks.append(temp.strip())
                    temp = sentence
                else:
                    temp = (temp + '. ' + sentence) if temp else sentence
            if temp.strip():
                final_chunks.append(temp.strip())
        return final_chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # INCREASED batch size for speed
        batch_size = int(os.getenv('EMBED_BATCH', '32'))  # Default to 32 for speed
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
            convert_to_tensor=False,
        )
        return embeddings.astype('float32').tolist()
    
    def create_table(self, table_name: str | None = None):
        table_name = table_name or self.get_default_table_name()
        schema = pa.schema([
            ("id", pa.int64()),
            ("chunk_id", pa.int64()),
            ("content", pa.string()),
            ("embedding", pa.list_(pa.float32(), self.get_embedding_dim() or 384)),
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
    
    def add_document_from_chunks(self, *,
                                 chunks: list[str],
                                 embeddings: list[list[float]],
                                 user_id: str,
                                 access_level: str = "private",
                                 doc_type: str = "policy",
                                 table_name: str = "policy_documents",
                                 document_id: str,
                                 source: str | None = None) -> str | None:
        try:
            if not chunks or not embeddings or len(chunks) != len(embeddings):
                print("Chunks and embeddings must be non-empty and of equal length")
                return None

            upload_timestamp = datetime.now()
            data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                data.append({
                    "id": i,
                    "chunk_id": i,
                    "content": chunk,
                    "embedding": embedding,
                    "source": source or "",
                    "chunk_index": i,
                    "user_id": user_id,
                    "document_id": document_id,
                    "upload_timestamp": upload_timestamp,
                    "access_level": access_level,
                    "type": doc_type
                })

            table = self.create_table(table_name)
            table.add(data)
            print(f"Successfully added {len(chunks)} precomputed chunks for document {document_id}")
            return document_id
        except Exception as e:
            print(f"Error adding precomputed document: {e}")
            return None
    
    def search_similar_in_document(self, query: str, user_id: str, document_id: str, table_name: str = "policy_documents", limit: int = 6):
        try:
            query_embedding = self.embedding_model.encode([query], show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)[0].astype('float32').tolist()
            table = self.db.open_table(table_name)
            where_expr = f"user_id = '{user_id}' AND document_id = '{document_id}'"
            results = table.search(query_embedding).where(where_expr).limit(limit).to_pandas()
            return results.to_dict('records')
        except Exception as e:
            print(f"Error searching document {document_id}: {e}")
            return []