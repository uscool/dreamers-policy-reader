import os
import tempfile
from typing import List

import numpy as np
import requests
import time
from dotenv import load_dotenv
from flask import Flask, jsonify, request

from llm_processor import LLMProcessor
from vector_db import PolicyVectorDB


def compute_cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    a = np.asarray(vector_a, dtype=np.float32)
    b = np.asarray(vector_b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rank_top_k_indices(embeddings: List[List[float]], query_embedding: List[float], top_k: int = 6) -> List[int]:
    """Return indices of the top_k most similar embeddings to the query embedding."""
    scored_indices = []
    for index, emb in enumerate(embeddings):
        similarity = compute_cosine_similarity(emb, query_embedding)
        scored_indices.append((index, similarity))
    scored_indices.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored_indices[:top_k]]


load_dotenv()
app = Flask(__name__)

# Token can be provided via env; defaulting to the provided sample for convenience
EXPECTED_TOKEN = os.getenv(
    "HACKRX_TOKEN",
    "fbc6d34d08858901d26a10f1b8796c2e77577b24bf91053390f858a35af05df7",
)

# Global singletons to avoid per-request model initialization
VECTOR_DB_SINGLETON: PolicyVectorDB | None = None
LLM_SINGLETON: LLMProcessor | None = None


def get_vector_db() -> PolicyVectorDB:
    global VECTOR_DB_SINGLETON
    if VECTOR_DB_SINGLETON is None:
        VECTOR_DB_SINGLETON = PolicyVectorDB()
    return VECTOR_DB_SINGLETON


def get_llm() -> LLMProcessor:
    global LLM_SINGLETON
    if LLM_SINGLETON is None:
        LLM_SINGLETON = LLMProcessor()
    return LLM_SINGLETON

# Eagerly warm up models on import (reduces first-request latency)
try:
    _ = get_vector_db()
    _ = get_llm()
except Exception:
    pass


def download_pdf_to_temp(url: str, connect_timeout: int, read_timeout: int, max_retries: int = 3) -> str:
    """Download a PDF from URL to a temporary file with retries and larger timeouts."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_path = tmp.name
    tmp.close()
    headers = {"User-Agent": "PolicyParse/1.0 (+https://example.local)"}
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(connect_timeout, read_timeout), headers=headers) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB
                        if chunk:
                            fh.write(chunk)
            return tmp_path
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                # Cleanup temp file on final failure
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                raise e


@app.post("/hackrx/run")
def hackrx_run():
    # Authorization (optional public mode)
    allow_public = os.getenv("ALLOW_PUBLIC", "false").lower() == "true"
    if not allow_public:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        token = auth_header.split(" ", 1)[1]
        if EXPECTED_TOKEN and token != EXPECTED_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401

    # Content-Type and payload
    if not request.is_json:
        return jsonify({"error": "Unsupported Media Type. Expected application/json."}), 415

    payload = request.get_json(silent=True) or {}
    document_url = payload.get("documents")
    questions = payload.get("questions")

    if not isinstance(document_url, str) or not document_url:
        return jsonify({"error": "'documents' must be a non-empty URL string"}), 400
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        return jsonify({"error": "'questions' must be a list of strings"}), 400

    # Download the PDF to a temporary file with retries
    try:
        temp_pdf_path = download_pdf_to_temp(
            document_url,
            connect_timeout=int(os.getenv("DOWNLOAD_CONNECT_TIMEOUT", "10")),
            read_timeout=int(os.getenv("DOWNLOAD_READ_TIMEOUT", "120")),
            max_retries=int(os.getenv("DOWNLOAD_RETRIES", "3")),
        )
    except Exception as download_error:
        return jsonify({"error": f"Failed to download document: {str(download_error)}"}), 400

    vector_db = get_vector_db()

    # PUBLIC user cache by URL: deterministic document_id
    import hashlib
    public_user_id = os.getenv("PUBLIC_USER_ID", "public")
    deterministic_doc_id = hashlib.sha256(document_url.encode("utf-8")).hexdigest()

    # Check if already in LanceDB for public user by document_id or source URL
    try:
        table = vector_db.create_table("policy_documents")
        # Try to see if any rows exist for this user+doc id
        existing = table.to_pandas()
        existing = existing[(existing["user_id"] == public_user_id) & ((existing["document_id"] == deterministic_doc_id) | (existing["source"] == document_url))]
        if existing is None:
            existing = []
    except Exception:
        existing = []

    chunks = None
    chunk_embeddings = None
    cached_doc_available = False
    if isinstance(existing, list) or existing is None or (hasattr(existing, "empty") and existing.empty):
        # Not cached: extract, chunk and embed, then insert once
        try:
            markdown_content = vector_db.extract_document_content(temp_pdf_path)
        except Exception as extraction_error:
            try:
                os.unlink(temp_pdf_path)
            except Exception:
                pass
            return jsonify({"error": f"Failed to extract document content: {str(extraction_error)}"}), 500
        finally:
            try:
                os.unlink(temp_pdf_path)
            except Exception:
                pass

        if not markdown_content:
            return jsonify({"error": "No content could be extracted from the document"}), 400

        chunks = vector_db.chunk_text(markdown_content)
        if not chunks:
            return jsonify({"error": "Document produced no searchable chunks"}), 400

        chunk_embeddings = vector_db.create_embeddings(chunks)
        if not chunk_embeddings:
            return jsonify({"error": "Failed to compute embeddings for document"}), 500

        # Insert into LanceDB for public user once
        try:
            vector_db.add_document(
                pdf_path=temp_pdf_path,
                user_id=public_user_id,
                access_level="public",
                doc_type="policy",
                table_name="policy_documents",
                document_id=deterministic_doc_id,
                source_override=document_url,
            )
            cached_doc_available = True
        except Exception:
            # Proceed even if insert fails; answers can be generated from current process
            pass
    else:
        # Cached: fetch chunks from LanceDB for this doc to build context quickly
        try:
            doc_rows = existing.sort_values("chunk_index").to_dict("records")
            chunks = [row["content"] for row in doc_rows]
            chunk_embeddings = [row["embedding"] for row in doc_rows]
            cached_doc_available = True
        except Exception:
            # Fallback to reprocess if reading cached rows fails
            try:
                markdown_content = vector_db.extract_document_content(temp_pdf_path)
            finally:
                try:
                    os.unlink(temp_pdf_path)
                except Exception:
                    pass
            if not markdown_content:
                return jsonify({"error": "No content could be extracted from the document"}), 400
            chunks = vector_db.chunk_text(markdown_content)
            chunk_embeddings = vector_db.create_embeddings(chunks)

    # Prepare answering setup
    llm_processor = get_llm()
    answers: List[str] = []

    # Batch-encode questions to reduce overhead
    try:
        question_embeddings = vector_db.embedding_model.encode(questions)
    except Exception:
        question_embeddings = [vector_db.embedding_model.encode([q])[0] for q in questions]

    # Build stronger context per question
    for question, question_embedding in zip(questions, question_embeddings):
        try:
            if cached_doc_available:
                # Prefer LanceDB ANN search within this document for better recall
                top_rows = get_vector_db().search_similar_in_document(
                    query=question,
                    user_id=public_user_id,
                    document_id=deterministic_doc_id,
                    table_name="policy_documents",
                    limit=6,
                )
                context = "\n\n".join([row.get("content", "") for row in top_rows])
                if not context:
                    # Fallback to local cosine ranking if ANN returns nothing
                    top_indices = rank_top_k_indices(chunk_embeddings, question_embedding, top_k=6)
                    context = "\n\n".join(chunks[i] for i in top_indices)
            else:
                top_indices = rank_top_k_indices(chunk_embeddings, question_embedding, top_k=6)
                context = "\n\n".join(chunks[i] for i in top_indices)

            if llm_processor.is_available():
                # Enforce numeric precision and policy quoting
                prompt = (
                    "Answer with exact numbers, durations, limits, and definitions as stated. "
                    "Prefer quoting the policy text verbatim for numeric constraints. "
                    "If not in context, reply 'Information not found in available documents.'\n\n"
                ) + llm_processor.prompt_template.format(query=question, context=context)
                response = llm_processor.llm.invoke(prompt)
                answer_text = (response.content or "").strip()
            else:
                best_chunk = chunks[top_indices[0]] if top_indices else ""
                answer_text = (best_chunk[:280] + "...") if len(best_chunk) > 280 else best_chunk

            if not answer_text:
                answer_text = "Information not found in available documents."

            answers.append(answer_text)
        except Exception:
            answers.append("Information not found in available documents.")

    return jsonify({"answers": answers})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


