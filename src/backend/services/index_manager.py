import logging
import os
import json
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..models.schema import CaseItem
from ..services.vector_store import VectorStore, create_vector_store, FaissVectorStore
from ..services.embedding_service import EmbeddingService
from ..utils.data_loader import load_saved_cases, load_dataset

logger = logging.getLogger(__name__)


class IndexManager:
    """Thread-safe manager for vector indices."""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one instance exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(IndexManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the index manager.
        
        Args:
            config: Configuration parameters
        """
        # Skip initialization if already initialized
        with self._lock:
            if self._initialized:
                return
            
            # Default configuration
            self.config = {
                "index_dir": "data/indices",
                "cases_path": "data/cases.json",
                "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                "vector_store_type": "faiss",
                "context_index_name": "ctx",
                "response_index_name": "resp"
            }
            
            # Update with provided config
            if config:
                self.config.update(config)
            
            # Initialize attributes
            self.context_index = None
            self.response_index = None
            self.cases = []
            self.embedding_service = None
            self.manifest = {
                "version": "1.0.0",
                "created_at": time.time(),
                "embed_model": self.config["embed_model"],
                "vector_store_type": self.config["vector_store_type"],
                "context_index_name": self.config["context_index_name"],
                "response_index_name": self.config["response_index_name"],
                "case_count": 0
            }
            
            self._initialized = True
            logger.info("Initialized index manager")
    
    def is_initialized(self) -> bool:
        """Check if indices are initialized."""
        with self._lock:
            return (
                self.context_index is not None and 
                self.response_index is not None and 
                len(self.cases) > 0
            )
    
    def build_indices(self, cases: List[CaseItem], embedding_service: EmbeddingService) -> None:
        """
        Build indices from cases.
        
        Args:
            cases: List of CaseItem objects
            embedding_service: Service for generating embeddings
        """
        with self._lock:
            start_time = time.time()
            logger.info(f"Building indices from {len(cases)} cases")
            
            self.cases = cases
            self.embedding_service = embedding_service
            
            # Generate embeddings
            embeddings = embedding_service.embed_cases(cases)
            context_embeddings = embeddings["context"]
            response_sentence_embeddings = embeddings["response_sentences"]
            
            # Create context index
            self.context_index = create_vector_store(
                self.config["vector_store_type"],
                embedding_service.vector_dim
            )
            
            # Create payloads for context index
            context_payloads = []
            for i, case in enumerate(cases):
                context_payloads.append({
                    "case_id": case.id,
                    "length": len(case.context),
                    "hash": hash(case.context) % 10000000
                })
            
            # Upsert context embeddings
            self.context_index.upsert(context_embeddings, context_payloads)
            
            # Create response index
            self.response_index = create_vector_store(
                self.config["vector_store_type"],
                embedding_service.vector_dim
            )
            
            # Create payloads for response index
            response_payloads = []
            flat_response_embeddings = []
            
            for case_idx, case in enumerate(cases):
                case_embeddings = response_sentence_embeddings[case_idx]
                for sent_idx, sentence in enumerate(case.response_sentences):
                    if sent_idx < len(case_embeddings):
                        flat_response_embeddings.append(case_embeddings[sent_idx])
                        response_payloads.append({
                            "case_id": case.id,
                            "sent_id": sent_idx,
                            "start": sentence.start,
                            "end": sentence.end,
                            "text": sentence.text,
                            "score": 0.0  # Will be filled during search
                        })
            
            # Convert to numpy array
            flat_response_embeddings = np.array(flat_response_embeddings)
            
            # Upsert response embeddings
            self.response_index.upsert(flat_response_embeddings, response_payloads)
            
            # Update manifest
            self.manifest["created_at"] = time.time()
            self.manifest["case_count"] = len(cases)
            
            logger.info(f"Built indices in {time.time() - start_time:.2f}s")
    
    def save_indices(self) -> None:
        """Save indices to disk."""
        with self._lock:
            if not self.is_initialized():
                logger.error("Cannot save indices: not initialized")
                return
            
            start_time = time.time()
            logger.info("Saving indices to disk")
            
            # Create index directory if it doesn't exist
            os.makedirs(self.config["index_dir"], exist_ok=True)
            
            # Save indices
            context_index_path = os.path.join(self.config["index_dir"], self.config["context_index_name"])
            response_index_path = os.path.join(self.config["index_dir"], self.config["response_index_name"])
            
            self.context_index.save(context_index_path)
            self.response_index.save(response_index_path)
            
            # Save manifest
            manifest_path = os.path.join(self.config["index_dir"], "index_manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(self.manifest, f, indent=2)
            
            # Save cases
            with open(self.config["cases_path"], 'w') as f:
                json.dump([case.model_dump() for case in self.cases], f)
            
            logger.info(f"Saved indices in {time.time() - start_time:.2f}s")
    
    def load_indices(self, embedding_service: Optional[EmbeddingService] = None) -> bool:
        """
        Load indices from disk.
        
        Args:
            embedding_service: Optional embedding service to use
            
        Returns:
            True if indices were loaded successfully, False otherwise
        """
        with self._lock:
            start_time = time.time()
            logger.info("Loading indices from disk")
            
            # Check if manifest exists
            manifest_path = os.path.join(self.config["index_dir"], "index_manifest.json")
            if not os.path.exists(manifest_path):
                logger.error(f"Manifest not found at {manifest_path}")
                return False
            
            # Load manifest
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            
            # Create embedding service if not provided
            if embedding_service is None:
                self.embedding_service = EmbeddingService(self.manifest["embed_model"])
            else:
                self.embedding_service = embedding_service
            
            # Create and load indices
            vector_dim = self.embedding_service.vector_dim
            
            # Check if cases exist
            if os.path.exists(self.config["cases_path"]):
                # Load cases from saved file
                try:
                    with open(self.config["cases_path"], 'r') as f:
                        case_data = json.load(f)
                    self.cases = [CaseItem(**case) for case in case_data]
                    logger.info(f"Loaded {len(self.cases)} cases from {self.config['cases_path']}")
                except Exception as e:
                    logger.error(f"Error loading cases from {self.config['cases_path']}: {e}")
                    # Try loading from dataset instead
                    dataset_path = os.path.join(os.path.dirname(self.config["cases_path"]), "combined_dataset.json")
                    if os.path.exists(dataset_path):
                        logger.info(f"Loading cases from dataset {dataset_path}")
                        self.cases = load_dataset(dataset_path)
                    else:
                        logger.error(f"No dataset found at {dataset_path}")
                        return False
            else:
                # Try loading from dataset
                dataset_path = os.path.join(os.path.dirname(self.config["cases_path"]), "combined_dataset.json")
                if os.path.exists(dataset_path):
                    logger.info(f"Loading cases from dataset {dataset_path}")
                    self.cases = load_dataset(dataset_path)
                else:
                    logger.error(f"No cases or dataset found")
                    return False
            
            # Load context index
            try:
                context_index_path = os.path.join(self.config["index_dir"], self.config["context_index_name"])
                self.context_index = FaissVectorStore(dim=vector_dim)
                
                # Check if the file exists with .index extension
                if os.path.exists(f"{context_index_path}.index"):
                    self.context_index.load(f"{context_index_path}.index")
                elif os.path.exists(context_index_path):
                    self.context_index.load(context_index_path)
                else:
                    logger.error(f"Context index not found at {context_index_path}")
                    return False
                
                # Load response index
                response_index_path = os.path.join(self.config["index_dir"], self.config["response_index_name"])
                self.response_index = FaissVectorStore(dim=vector_dim)
                
                # Check if the file exists with .index extension
                if os.path.exists(f"{response_index_path}.index"):
                    self.response_index.load(f"{response_index_path}.index")
                elif os.path.exists(response_index_path):
                    self.response_index.load(response_index_path)
                else:
                    logger.error(f"Response index not found at {response_index_path}")
                    return False
                
                logger.info(f"Loaded indices in {time.time() - start_time:.2f}s")
                logger.info(f"Context index: {self.context_index.count()} vectors")
                logger.info(f"Response index: {self.response_index.count()} vectors")
                
                return True
            except Exception as e:
                logger.error(f"Error loading indices: {e}")
                return False
    
    def get_indices(self) -> Tuple[VectorStore, VectorStore, List[CaseItem], EmbeddingService]:
        """
        Get the indices and related objects.
        
        Returns:
            Tuple of (context_index, response_index, cases, embedding_service)
        """
        with self._lock:
            if not self.is_initialized():
                raise ValueError("Indices not initialized")
            
            return self.context_index, self.response_index, self.cases, self.embedding_service
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indices.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            if not self.is_initialized():
                return {"initialized": False}
            
            return {
                "initialized": True,
                "case_count": len(self.cases),
                "context_index_count": self.context_index.count(),
                "response_index_count": self.response_index.count(),
                "embed_model": self.manifest["embed_model"],
                "vector_store_type": self.manifest["vector_store_type"],
                "created_at": self.manifest["created_at"]
            } 