import os
import math
import re
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from app.models import UploadedFile

# Simple fallback in-memory database for text chunks when ChromaDB is not used
# Format: {session_id: [{"text": str, "file_name": str, "chunk_index": int}]}
_fallback_vector_store: Dict[str, List[Dict[str, Any]]] = {}


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks of a target size."""
    words = text.split()
    chunks = []
    
    # Simple split based on word count
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
        i += chunk_size - chunk_overlap
        
    return chunks


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file using PyPDF."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def extract_text_from_excel(file_path: str) -> str:
    """Extract readable text representing tables from an Excel file."""
    import pandas as pd
    try:
        xls = pd.ExcelFile(file_path)
        text_parts = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.empty:
                continue
            text_parts.append(f"Sheet Name: {sheet_name}")
            # Convert to tab-separated string representation for cleaner reading
            csv_str = df.to_csv(index=False, sep="\t")
            text_parts.append(csv_str)
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"⚠️ Error extracting Excel: {e}")
        return ""


def extract_text_from_csv(file_path: str) -> str:
    """Extract text from a CSV file."""
    import pandas as pd
    try:
        df = pd.read_csv(file_path)
        return df.to_csv(index=False, sep="\t")
    except Exception as e:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""


def extract_text_from_file(file_path: str, mime_type: str) -> str:
    """Extract raw text from PDF, Excel, CSV, or TXT files."""
    mime_lower = mime_type.lower()
    file_path_lower = file_path.lower()
    
    if "pdf" in mime_lower or file_path_lower.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif "excel" in mime_lower or "spreadsheet" in mime_lower or file_path_lower.endswith(".xlsx") or file_path_lower.endswith(".xls"):
        return extract_text_from_excel(file_path)
    elif "csv" in mime_lower or file_path_lower.endswith(".csv"):
        return extract_text_from_csv(file_path)
    else:
        # Assume plain text / markdown
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def index_document(file_id: str, session_id: str, file_name: str, file_path: str, mime_type: str):
    """Parse document and index its contents into vector store / fallback system."""
    try:
        raw_text = extract_text_from_file(file_path, mime_type)
        cleaned = clean_text(raw_text)
        if not cleaned:
            print(f"⚠️ Warning: File {file_name} was empty or could not be parsed.")
            return

        chunks = chunk_text(cleaned)
        print(f"ℹ️ Chunked '{file_name}' into {len(chunks)} fragments.")

        try:
            import chromadb
            chroma_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
            os.makedirs(chroma_dir, exist_ok=True)
            
            client = chromadb.PersistentClient(path=chroma_dir)
            collection = client.get_or_create_collection(
                name=f"session_{session_id.replace('-', '_')}"
            )
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({"file_name": file_name, "file_id": file_id})
                ids.append(f"{file_id}_{idx}")
                
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Indexed {len(chunks)} chunks into ChromaDB for session {session_id}.")
            return
        except Exception as chroma_err:
            print(f"⚠️ ChromaDB indexing failed ({str(chroma_err)}). Falling back to pure Python TF-IDF engine.")

        # Fallback index in memory
        if session_id not in _fallback_vector_store:
            _fallback_vector_store[session_id] = []

        for idx, chunk in enumerate(chunks):
            _fallback_vector_store[session_id].append({
                "text": chunk,
                "file_name": file_name,
                "file_id": file_id,
                "chunk_index": idx
            })
        print(f"✅ Cached {len(chunks)} chunks in memory for session {session_id}.")

    except Exception as e:
        print(f"❌ Error indexing document {file_name}: {str(e)}")
        raise e


# Helper functions for lightweight TF-IDF and Cosine Similarity implementation
def tokenize(text: str) -> List[str]:
    """Tokenize and convert text to lowercase."""
    return re.findall(r'\b\w{3,15}\b', text.lower())


def query_rag(session_id: str, query: str, top_k: int = 3) -> str:
    """Query document store (ChromaDB or fallback) to search for segments relevant to the query."""
    if not query:
        return ""

    # Try ChromaDB first
    try:
        import chromadb
        chroma_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
        if os.path.exists(chroma_dir):
            client = chromadb.PersistentClient(path=chroma_dir)
            # Check if collection exists
            collection_name = f"session_{session_id.replace('-', '_')}"
            
            # Simple check for collection existence via listing
            collections = [c.name for c in client.list_collections()]
            if collection_name in collections:
                collection = client.get_collection(name=collection_name)
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
                
                # Format results
                if results and 'documents' in results and results['documents']:
                    docs = results['documents'][0]
                    sources = results['metadatas'][0] if 'metadatas' in results else []
                    
                    formatted_results = []
                    for idx, doc in enumerate(docs):
                        source_name = sources[idx].get("file_name", "Unknown File") if idx < len(sources) else "Document"
                        formatted_results.append(f"--- Context Segment {idx+1} (Source: {source_name}) ---\n{doc}")
                    
                    return "\n\n".join(formatted_results)
    except Exception as chroma_err:
        print(f"⚠️ ChromaDB query failed ({str(chroma_err)}). Trying pure Python TF-IDF search.")

    # Fallback similarity search: custom TF-IDF in pure Python
    # NOTE: This lightweight TF-IDF implementation is used as a fallback
    # for the demonstration stand to avoid heavy dependencies (scikit-learn, numpy).
    # In a production environment, use pgvector, Elasticsearch, or ChromaDB.
    session_docs = _fallback_vector_store.get(session_id, [])
    if not session_docs:
        return ""

    query_tokens = tokenize(query)
    if not query_tokens:
        return ""

    doc_count = len(session_docs)
    df = {}
    for doc in session_docs:
        tokens = set(tokenize(doc["text"]))
        for t in tokens:
            df[t] = df.get(t, 0) + 1

    idf = {}
    for t, count in df.items():
        idf[t] = math.log((1 + doc_count) / (1 + count)) + 1

    scored_docs = []
    for doc in session_docs:
        doc_tokens = tokenize(doc["text"])
        if not doc_tokens:
            continue
        
        tf = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1
            
        dot_product = 0.0
        query_norm = 0.0
        doc_norm = 0.0
        
        # Intersection of query and document tokens
        for q_token in set(query_tokens):
            w_q = query_tokens.count(q_token) * idf.get(q_token, 1.0)
            w_d = tf.get(q_token, 0) * idf.get(q_token, 1.0)
            dot_product += w_q * w_d
            query_norm += w_q ** 2

        for t, count in tf.items():
            doc_norm += (count * idf.get(t, 1.0)) ** 2

        if query_norm > 0 and doc_norm > 0:
            similarity = dot_product / (math.sqrt(query_norm) * math.sqrt(doc_norm))
        else:
            similarity = 0.0

        if similarity > 0:
            scored_docs.append((similarity, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_docs[:top_k]

    if not top_matches:
        top_matches = [(0.0, doc) for doc in session_docs[:top_k]]

    formatted_results = []
    for idx, (score, doc) in enumerate(top_matches):
        formatted_results.append(
            f"--- Context Segment {idx+1} (Source: {doc['file_name']}) ---\n{doc['text']}"
        )

    return "\n\n".join(formatted_results)
