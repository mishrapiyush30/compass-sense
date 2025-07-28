from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
import faiss
import os
import json
import logging

logger = logging.getLogger(__name__)


class Hit:
    """Represents a hit from vector search."""
    def __init__(self, id: int, score: float, payload: Dict[str, Any]):
        self.id = id
        self.score = score
        self.payload = payload


class VectorStore(ABC):
    """Abstract interface for vector stores."""
    
    @abstractmethod
    def upsert(self, vectors: np.ndarray, payloads: List[Dict[str, Any]]) -> None:
        """Insert or update vectors with their payloads."""
        pass
    
    @abstractmethod
    def search(self, qvec: np.ndarray, k: int, filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Hit]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors in the store."""
        pass
    
    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return stats about the vector store."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        pass


class FaissVectorStore(VectorStore):
    """FAISS implementation of VectorStore."""
    
    def __init__(self, dim: int, metric: str = "ip"):
        """
        Initialize a FAISS vector store.
        
        Args:
            dim: Dimension of vectors
            metric: Distance metric ('ip' for inner product, 'l2' for L2 distance)
        """
        self.dim = dim
        self.metric = metric
        
        if metric == "ip":
            self.index = faiss.IndexFlatIP(dim)
        elif metric == "l2":
            self.index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        self.payloads = []
        self.manifest = {
            "type": "faiss",
            "dim": dim,
            "metric": metric,
            "count": 0,
            "version": "1.0.0"
        }
    
    def upsert(self, vectors: np.ndarray, payloads: List[Dict[str, Any]]) -> None:
        """Insert vectors and payloads."""
        if len(vectors) != len(payloads):
            raise ValueError("Number of vectors and payloads must match")
        
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors of dimension {self.dim}, got {vectors.shape[1]}")
        
        # Convert to float32 if needed
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        # Add to index
        self.index.add(vectors)
        self.payloads.extend(payloads)
        self.manifest["count"] = len(self.payloads)
    
    def search(self, qvec: np.ndarray, k: int, filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Hit]:
        """Search for similar vectors."""
        if qvec.shape[0] != self.dim:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dim}, got {qvec.shape[0]}")
        
        # Ensure qvec is float32 and reshape if needed
        qvec = qvec.astype(np.float32).reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(qvec, k)
        
        # Create hits
        hits = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx != -1:  # FAISS returns -1 for not enough results
                payload = self.payloads[idx]
                
                # Apply filter if provided
                if filter_fn is None or filter_fn(payload):
                    hits.append(Hit(id=idx, score=float(score), payload=payload))
        
        return hits
    
    def count(self) -> int:
        """Return the number of vectors in the store."""
        return len(self.payloads)
    
    def stats(self) -> Dict[str, Any]:
        """Return stats about the vector store."""
        return {
            "type": "faiss",
            "dim": self.dim,
            "metric": self.metric,
            "count": len(self.payloads)
        }
    
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save index
        index_path = f"{path}.index"
        faiss.write_index(self.index, index_path)
        
        # Save payloads
        payloads_path = f"{path}.payloads.json"
        with open(payloads_path, 'w') as f:
            json.dump(self.payloads, f)
        
        # Save manifest
        manifest_path = f"{path}.manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f)
        
        logger.info(f"Saved FAISS index to {index_path} with {len(self.payloads)} vectors")
    
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        # Check if the index file exists directly
        if os.path.exists(path):
            # Direct file path provided
            self.index = faiss.read_index(path)
            
            payloads_path = None

            # Try the exact pair written by save(): *.payloads.json
            cand = path.replace('.index', '.payloads.json')
            if os.path.exists(cand):
                payloads_path = cand
            else:
                # legacy/alternative names
                cand2 = path.replace('.index', '_payloads.json')
                if os.path.exists(cand2):
                    payloads_path = cand2
                else:
                    base_name = os.path.basename(path).split('.')[0]
                    dir_name = os.path.dirname(path)
                    for alt in (f"{base_name}_payloads.json",
                                os.path.join(dir_name, "context_payloads.json"),
                                os.path.join(dir_name, "sentence_payloads.json")):
                        if os.path.exists(alt):
                            payloads_path = alt
                            break

            if payloads_path and os.path.exists(payloads_path):
                with open(payloads_path, 'r') as f:
                    self.payloads = json.load(f)
                logger.info(f"Loaded payloads from {payloads_path}")
            else:
                logger.warning(f"No payloads found for {path}")
                self.payloads = []
            
            logger.info(f"Loaded FAISS index from {path} with {len(self.payloads)} payloads")
            return
        
        # Traditional path format with extensions
        index_path = f"{path}.index"
        if not os.path.exists(index_path):
            # Try without the extension
            index_path = path
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index file not found at {path}.index or {path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load payloads
        payloads_path = f"{path}.payloads.json"
        if not os.path.exists(payloads_path):
            # Try alternative paths
            base_name = os.path.basename(path)
            dir_name = os.path.dirname(path)
            
            if "ctx" in base_name:
                payloads_path = os.path.join(dir_name, "context_payloads.json")
            elif "resp" in base_name:
                payloads_path = os.path.join(dir_name, "sentence_payloads.json")
        
        if os.path.exists(payloads_path):
            with open(payloads_path, 'r') as f:
                self.payloads = json.load(f)
        else:
            logger.warning(f"Payloads file not found at {payloads_path}")
            self.payloads = []
        
        # Load manifest
        manifest_path = f"{path}.manifest.json"
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
        else:
            # Try to load from index_manifest.json
            manifest_path = os.path.join(os.path.dirname(path), "index_manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                    self.manifest = {
                        "type": "faiss",
                        "dim": manifest.get("vector_dim", self.dim),
                        "metric": "ip",
                        "count": len(self.payloads),
                        "version": manifest.get("version", "1.0.0")
                    }
            else:
                logger.warning(f"Manifest file not found at {manifest_path}")
        
        logger.info(f"Loaded FAISS index from {index_path} with {len(self.payloads)} vectors")


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of VectorStore (placeholder for production)."""
    
    def __init__(self, dim: int, collection_name: str, url: str = "http://localhost:6333"):
        """
        Initialize a Qdrant vector store.
        
        Args:
            dim: Dimension of vectors
            collection_name: Name of the collection
            url: URL of the Qdrant server
        """
        self.dim = dim
        self.collection_name = collection_name
        self.url = url
        self.manifest = {
            "type": "qdrant",
            "dim": dim,
            "collection": collection_name,
            "url": url,
            "version": "1.0.0"
        }
        
        # In a real implementation, we would initialize the Qdrant client here
        # For now, we'll just log that this is a placeholder
        logger.info("QdrantVectorStore is a placeholder. In production, implement with the Qdrant client.")
    
    def upsert(self, vectors: np.ndarray, payloads: List[Dict[str, Any]]) -> None:
        """Insert or update vectors with their payloads."""
        # Placeholder for Qdrant implementation
        logger.info(f"Would upsert {len(vectors)} vectors to Qdrant collection {self.collection_name}")
    
    def search(self, qvec: np.ndarray, k: int, filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Hit]:
        """Search for similar vectors."""
        # Placeholder for Qdrant implementation
        logger.info(f"Would search for {k} similar vectors in Qdrant collection {self.collection_name}")
        return []
    
    def count(self) -> int:
        """Return the number of vectors in the store."""
        # Placeholder for Qdrant implementation
        return 0
    
    def stats(self) -> Dict[str, Any]:
        """Return stats about the vector store."""
        return self.manifest
    
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        # For Qdrant, we would just save the manifest as the data is stored in the Qdrant server
        manifest_path = f"{path}.manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f)
        
        logger.info(f"Saved Qdrant manifest to {manifest_path}")
    
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        # For Qdrant, we would just load the manifest
        manifest_path = f"{path}.manifest.json"
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        logger.info(f"Loaded Qdrant manifest from {manifest_path}")


def create_vector_store(store_type: str, dim: int, **kwargs) -> VectorStore:
    """Factory function to create a vector store."""
    if store_type == "faiss":
        metric = kwargs.get("metric", "ip")
        return FaissVectorStore(dim=dim, metric=metric)
    elif store_type == "qdrant":
        collection_name = kwargs.get("collection_name", "default")
        url = kwargs.get("url", "http://localhost:6333")
        return QdrantVectorStore(dim=dim, collection_name=collection_name, url=url)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}") 