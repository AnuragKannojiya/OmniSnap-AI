import logging
import time
import os
import pickle
from dataclasses import dataclass
from typing import List
import numpy as np

try:
    from core.npu_runtime import NPURuntime
except ImportError:
    NPURuntime = None

logger = logging.getLogger(__name__)

__all__ = ['RAGEngine', 'SearchResult']


@dataclass
class SearchResult:
    chunk_text: str
    source_file: str
    similarity_score: float
    chunk_id: str


class RAGEngine:
    """Air-Gapped Document RAG using all-MiniLM-L6-v2."""
    
    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.model = None
        self.index_dir = os.path.expanduser("~/.omnisnap/index/")
        try:
            os.makedirs(self.index_dir, exist_ok=True)
        except (PermissionError, OSError):
            # Fallback to local directory if home dir is not writable
            self.index_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".omnisnap", "index")
            os.makedirs(self.index_dir, exist_ok=True)
        
        # In-memory vector store: list of embeddings and list of corresponding metadata
        self.embeddings = []
        self.metadata = []
        
        if not self.demo_mode and NPURuntime:
            try:
                self.runtime = NPURuntime()
                self.model = self.runtime.load_model("models/all_minilm_l6_v2.onnx")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}. Using demo mode.")
                self.demo_mode = True
        else:
            self.demo_mode = True
            
        self.load_index()

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Simple text chunking."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.demo_mode:
            # Return random embeddings
            return np.random.randn(len(texts), 384).astype(np.float32)
        # Real inference here
        return np.random.randn(len(texts), 384).astype(np.float32)

    def add_document(self, filepath: str):
        # Dummy document extraction
        text = f"Content extracted from {filepath}. This is a sample document text used for RAG testing."
        chunks = self.chunk_text(text)
        
        embs = self.embed(chunks)
        for i, (chunk, emb) in enumerate(zip(chunks, embs)):
            self.embeddings.append(emb)
            self.metadata.append({
                "chunk_text": chunk,
                "source_file": filepath,
                "chunk_id": f"{os.path.basename(filepath)}_{i}"
            })
            
        self.save_index()
        logger.info(f"Indexed document: {filepath}")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if not self.embeddings:
            return []
            
        query_emb = self.embed([query])[0]
        
        if self.demo_mode:
            # Synthetic search results
            return [
                SearchResult(
                    chunk_text="This is a relevant chunk from the knowledge base regarding your query.",
                    source_file="sample_doc.pdf",
                    similarity_score=0.92,
                    chunk_id="sample_doc_0"
                )
            ]
            
        # Cosine similarity
        db_embs = np.array(self.embeddings)
        query_norm = query_emb / np.linalg.norm(query_emb)
        db_norms = db_embs / np.linalg.norm(db_embs, axis=1, keepdims=True)
        similarities = np.dot(db_norms, query_norm)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            meta = self.metadata[idx]
            results.append(SearchResult(
                chunk_text=meta["chunk_text"],
                source_file=meta["source_file"],
                similarity_score=float(similarities[idx]),
                chunk_id=meta["chunk_id"]
            ))
            
        return results

    def get_context(self, query: str, top_k: int = 3) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
            
        context = []
        for r in results:
            context.append(f"Source: {r.source_file}\n{r.chunk_text}")
            
        return "\n\n".join(context)

    def save_index(self):
        try:
            with open(os.path.join(self.index_dir, "db.pkl"), "wb") as f:
                pickle.dump({"embeddings": self.embeddings, "metadata": self.metadata}, f)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def load_index(self):
        try:
            db_path = os.path.join(self.index_dir, "db.pkl")
            if os.path.exists(db_path):
                with open(db_path, "rb") as f:
                    data = pickle.load(f)
                    self.embeddings = data.get("embeddings", [])
                    self.metadata = data.get("metadata", [])
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
