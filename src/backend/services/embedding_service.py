import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import os
import time

from ..models.schema import CaseItem, SentenceSpan

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        logger.info(f"Loading embedding model: {model_name}")
        start_time = time.time()
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model in {time.time() - start_time:.2f}s. Vector dimension: {self.vector_dim}")
    
    def embed_text(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed (string or list of strings)
            normalize: Whether to normalize the embeddings to unit length
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(text, str):
            text = [text]
        
        # Generate embeddings
        embeddings = self.model.encode(text, normalize_embeddings=normalize)
        
        return embeddings
    
    def embed_cases(self, cases: List[CaseItem], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for cases.
        
        Args:
            cases: List of CaseItem objects
            batch_size: Batch size for embedding generation
            
        Returns:
            Dictionary with 'context' and 'response_sentences' embeddings
        """
        logger.info(f"Generating embeddings for {len(cases)} cases")
        start_time = time.time()
        
        # Extract contexts and response sentences
        contexts = [case.context for case in cases]
        
        # Flatten response sentences for batch processing
        response_sentences = []
        sentence_case_map = []  # To keep track of which case each sentence belongs to
        
        for case_idx, case in enumerate(cases):
            for sent in case.response_sentences:
                response_sentences.append(sent.text)
                sentence_case_map.append(case_idx)
        
        # Generate embeddings in batches
        logger.info(f"Embedding {len(contexts)} contexts")
        context_embeddings = self.embed_text(contexts)
        
        logger.info(f"Embedding {len(response_sentences)} response sentences")
        response_sentence_embeddings = self.embed_text(response_sentences)
        
        # Group response sentence embeddings by case
        case_response_embeddings = [[] for _ in range(len(cases))]
        for sent_idx, case_idx in enumerate(sentence_case_map):
            case_response_embeddings[case_idx].append(response_sentence_embeddings[sent_idx])
        
        # Convert to numpy arrays
        case_response_embeddings = [np.array(embs) for embs in case_response_embeddings]
        
        logger.info(f"Generated embeddings in {time.time() - start_time:.2f}s")
        
        return {
            "context": context_embeddings,
            "response_sentences": case_response_embeddings
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "vector_dim": self.vector_dim
        } 