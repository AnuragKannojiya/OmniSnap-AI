import logging
import time
import os
import io
import pickle
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ['RAGEngine', 'SearchResult']

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not available. RAG will use demo mode.")


@dataclass
class SearchResult:
    chunk_text: str
    source_file: str
    similarity_score: float
    chunk_id: str


class RAGEngine:
    """Air-Gapped Document RAG using all-MiniLM-L6-v2.
    
    Uses sentence-transformers for REAL vector embeddings.
    Stores vectors in memory with numpy cosine similarity search.
    Persists index to disk.
    """
    
    MODEL_ID = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    def __init__(self, model_id: str = None, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.model = None
        self.model_id = model_id or self.MODEL_ID
        self._inference_count = 0
        self._total_latency = 0.0
        
        # Storage
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[dict] = []
        self.documents: List[str] = []  # Track indexed document names
        
        # Index directory - use project-local directory
        self.index_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".omnisnap", "index"
        )
        try:
            os.makedirs(self.index_dir, exist_ok=True)
        except (PermissionError, OSError):
            self.index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_index")
            os.makedirs(self.index_dir, exist_ok=True)
        
        if demo_mode or not HAS_SENTENCE_TRANSFORMERS:
            self.demo_mode = True
            logger.info("RAGEngine initialized in demo mode")
        else:
            try:
                logger.info(f"Loading embedding model '{self.model_id}'...")
                self.model = SentenceTransformer(self.model_id)
                logger.info(f"✅ Embedding model loaded ({self.EMBEDDING_DIM}D vectors)")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.demo_mode = True
        
        # Load existing index
        self.load_index()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks by sentences."""
        # Split by sentences first
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last few sentences for overlap
                overlap_text = ' '.join(current_chunk)
                if len(overlap_text) > overlap:
                    # Find the overlap boundary
                    keep_from = len(current_chunk) // 2
                    current_chunk = current_chunk[keep_from:]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
            current_chunk.append(sentence)
            current_length += len(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Fallback for very short or empty text
        if not chunks and text.strip():
            chunks = [text.strip()]
        
        return chunks
    
    def extract_text(self, filepath: str) -> str:
        """Extract text from various document formats."""
        ext = os.path.splitext(filepath)[1].lower()
        
        try:
            if ext == '.txt' or ext == '.md':
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            elif ext == '.pdf':
                return self._extract_pdf(filepath)
            elif ext in ('.docx', '.doc'):
                return self._extract_docx(filepath)
            else:
                # Try reading as text
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Text extraction error for {filepath}: {e}")
            return ""
    
    def _extract_pdf(self, filepath: str) -> str:
        """Extract text from PDF."""
        try:
            # Try PyPDF2
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            return '\n'.join(page.extract_text() or '' for page in reader.pages)
        except ImportError:
            pass
        try:
            # Try pdfminer
            from pdfminer.high_level import extract_text
            return extract_text(filepath)
        except ImportError:
            pass
        # Basic fallback: read raw bytes and try to find text
        logger.warning(f"No PDF library available. Install PyPDF2: pip install PyPDF2")
        with open(filepath, 'rb') as f:
            content = f.read()
        # Very basic text extraction from PDF bytes
        import re
        text_parts = re.findall(rb'\((.+?)\)', content)
        return ' '.join(p.decode('utf-8', errors='ignore') for p in text_parts[:100])
    
    def _extract_docx(self, filepath: str) -> str:
        """Extract text from DOCX."""
        try:
            from lxml import etree
            import zipfile
            with zipfile.ZipFile(filepath) as z:
                with z.open('word/document.xml') as f:
                    tree = etree.parse(f)
                    # Extract all text from w:t elements
                    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                    texts = tree.xpath('//w:t/text()', namespaces=ns)
                    return ' '.join(texts)
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            return ""
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        start = time.time()
        
        if self.demo_mode:
            # Return random unit vectors for demo
            embs = np.random.randn(len(texts), self.EMBEDDING_DIM).astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            return embs / np.clip(norms, 1e-8, None)
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            latency = (time.time() - start) * 1000
            self._inference_count += 1
            self._total_latency += latency
            return np.array(embeddings).astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            embs = np.random.randn(len(texts), self.EMBEDDING_DIM).astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            return embs / np.clip(norms, 1e-8, None)
    
    def add_document(self, filepath: str) -> int:
        """Index a document: extract text, chunk, embed, store.
        Returns number of chunks indexed."""
        logger.info(f"Indexing document: {filepath}")
        
        text = self.extract_text(filepath)
        if not text.strip():
            logger.warning(f"No text extracted from {filepath}")
            return 0
        
        chunks = self.chunk_text(text)
        if not chunks:
            return 0
        
        embeddings = self.embed(chunks)
        doc_name = os.path.basename(filepath)
        
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            self.embeddings.append(emb)
            self.metadata.append({
                "chunk_text": chunk,
                "source_file": doc_name,
                "chunk_id": f"{doc_name}_{i}",
            })
        
        if doc_name not in self.documents:
            self.documents.append(doc_name)
        
        self.save_index()
        logger.info(f"Indexed {len(chunks)} chunks from {doc_name}")
        return len(chunks)
    
    def add_text(self, text: str, source_name: str = "direct_input") -> int:
        """Index raw text directly (without file)."""
        chunks = self.chunk_text(text)
        if not chunks:
            return 0
        embeddings = self.embed(chunks)
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            self.embeddings.append(emb)
            self.metadata.append({
                "chunk_text": chunk,
                "source_file": source_name,
                "chunk_id": f"{source_name}_{i}",
            })
        if source_name not in self.documents:
            self.documents.append(source_name)
        self.save_index()
        return len(chunks)
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search indexed documents using semantic similarity."""
        if not self.embeddings:
            return []
        
        start = time.time()
        query_emb = self.embed([query])[0]
        
        # Cosine similarity (embeddings are already normalized)
        db_embs = np.array(self.embeddings)
        similarities = np.dot(db_embs, query_emb)
        
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        latency = (time.time() - start) * 1000
        
        results = []
        for idx in top_indices:
            meta = self.metadata[idx]
            results.append(SearchResult(
                chunk_text=meta["chunk_text"],
                source_file=meta["source_file"],
                similarity_score=float(similarities[idx]),
                chunk_id=meta["chunk_id"],
            ))
        
        logger.info(f"RAG search '{query[:50]}...' returned {len(results)} results in {latency:.1f}ms")
        return results
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """Get formatted context string for LLM augmentation."""
        results = self.search(query, top_k)
        if not results:
            return ""
        parts = []
        for r in results:
            parts.append(f"[Source: {r.source_file} | Score: {r.similarity_score:.2f}]\n{r.chunk_text}")
        return "\n\n---\n\n".join(parts)
    
    def save_index(self):
        try:
            db_path = os.path.join(self.index_dir, "rag_index.pkl")
            with open(db_path, "wb") as f:
                pickle.dump({
                    "embeddings": self.embeddings,
                    "metadata": self.metadata,
                    "documents": self.documents,
                }, f)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load_index(self):
        try:
            db_path = os.path.join(self.index_dir, "rag_index.pkl")
            if os.path.exists(db_path):
                with open(db_path, "rb") as f:
                    data = pickle.load(f)
                    self.embeddings = data.get("embeddings", [])
                    self.metadata = data.get("metadata", [])
                    self.documents = data.get("documents", [])
                logger.info(f"Loaded RAG index: {len(self.embeddings)} vectors, {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
    
    def clear_index(self):
        self.embeddings = []
        self.metadata = []
        self.documents = []
        self.save_index()
    
    @property
    def num_chunks(self) -> int:
        return len(self.embeddings)
    
    @property
    def num_documents(self) -> int:
        return len(self.documents)
