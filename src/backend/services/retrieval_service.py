import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from rank_bm25 import BM25Okapi
import re

from ..models.schema import CaseItem, SentenceSpan, SearchResult, Highlight
from ..services.vector_store import VectorStore, Hit
from ..services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for retrieving cases and quotable sentences."""
    
    def __init__(
        self,
        context_index: VectorStore,
        response_index: VectorStore,
        embedding_service: EmbeddingService,
        cases: List[CaseItem],
        config: Dict[str, Any] = None
    ):
        """
        Initialize the retrieval service.
        
        Args:
            context_index: Vector store for context embeddings
            response_index: Vector store for response sentence embeddings
            embedding_service: Service for generating embeddings
            cases: List of all cases
            config: Configuration parameters
        """
        self.context_index = context_index
        self.response_index = response_index
        self.embedding_service = embedding_service
        self.cases = cases
        
        # Default configuration
        self.config = {
            "k1": 100,              # Number of cases to retrieve in first stage (increased from 30)
            "n": 10,                # Number of cases to probe for sentences
            "m": 3,                 # Number of sentences to retrieve per case
            "top_final": 3,         # Number of final cases to return
            "rrf_c": 60,            # RRF constant
            "mmr_lambda": 0.7,      # MMR lambda (balance between relevance and diversity)
            "min_score_threshold": 0.01  # Minimum score threshold for retrieval
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Initialize BM25 for lexical search
        self._init_bm25()
        
        logger.info(f"Initialized retrieval service with {len(cases)} cases")
    
    def _init_bm25(self):
        """Initialize BM25 for lexical search."""
        # Tokenize contexts for BM25
        tokenized_contexts = []
        for case in self.cases:
            # Simple tokenization: lowercase, remove punctuation, split on whitespace
            tokens = re.sub(r'[^\w\s]', '', case.context.lower()).split()
            tokenized_contexts.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_contexts)
        logger.info("Initialized BM25 for lexical search")
    
    def _lexical_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Perform lexical search using BM25.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (case_id, score) tuples
        """
        # Tokenize query
        query_tokens = re.sub(r'[^\w\s]', '', query.lower()).split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices and scores
        top_idx = np.argsort(scores)[::-1][:k]
        top_scores = scores[top_idx]
        
        # Normalize scores to [0, 1]
        if top_scores.max() > 0:
            top_scores = top_scores / top_scores.max()
        
        # Map list index -> actual case_id
        results = []
        for idx, sc in zip(top_idx, top_scores):
            case_id = self.cases[int(idx)].id
            results.append((int(case_id), float(sc)))
        
        return results
    
    def _rrf_fusion(
        self, 
        dense_results: List[Tuple[int, float]], 
        lexical_results: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Perform RRF fusion of dense and lexical results.
        
        Args:
            dense_results: List of (case_id, score) tuples from dense search
            lexical_results: List of (case_id, score) tuples from lexical search
            
        Returns:
            List of (case_id, score) tuples with fused scores
        """
        c = self.config["rrf_c"]
        
        # Create dictionaries for easy lookup
        dense_dict = {case_id: (i, score) for i, (case_id, score) in enumerate(dense_results)}
        lexical_dict = {case_id: (i, score) for i, (case_id, score) in enumerate(lexical_results)}
        
        # Get all unique case IDs
        all_case_ids = set(dict(dense_results).keys()) | set(dict(lexical_results).keys())
        
        # Calculate RRF scores
        rrf_scores = []
        for case_id in all_case_ids:
            dense_rank = dense_dict.get(case_id, (len(dense_results), 0))[0] + 1  # 1-indexed
            lexical_rank = lexical_dict.get(case_id, (len(lexical_results), 0))[0] + 1  # 1-indexed
            
            # RRF formula: 1 / (c + rank)
            rrf_score = (1 / (c + dense_rank)) + (1 / (c + lexical_rank))
            rrf_scores.append((case_id, rrf_score))
        
        # Sort by RRF score (descending)
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        
        return rrf_scores
    
    def _mmr_diversification(
        self, 
        case_ids: List[int],
        case_scores: List[float],
        query_embedding: np.ndarray,
        k: int
    ) -> List[Tuple[int, float]]:
        """
        Perform MMR diversification.
        
        Args:
            case_ids: List of case IDs
            case_scores: List of case scores (cosine similarities)
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            List of (case_id, score) tuples with diversified ordering
        """
        lambda_param = self.config["mmr_lambda"]
        
        # Get case embeddings
        case_embeddings = []
        for case_id in case_ids:
            try:
                # Find the case in self.cases
                case = next((c for c in self.cases if c.id == case_id), None)
                if case is None:
                    logger.warning(f"Case ID {case_id} not found in cases list")
                    # Use a zero vector as fallback
                    case_embeddings.append(np.zeros(self.embedding_service.vector_dim))
                    continue
                
                # Get embedding
                embedding = self.embedding_service.embed_text(case.context)
                # Handle both single embedding and batch of embeddings
                if len(embedding.shape) > 1:
                    embedding = embedding[0]  # Take first embedding if it's a batch
                case_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error embedding case {case_id}: {e}")
                # Use a zero vector as fallback
                case_embeddings.append(np.zeros(self.embedding_service.vector_dim))
        
        case_embeddings = np.array(case_embeddings)
        
        # Initialize selected and remaining indices
        selected_indices = []
        remaining_indices = list(range(len(case_ids)))
        
        if not remaining_indices:
            logger.warning("No cases to diversify")
            return []
        
        # Select first item (highest score)
        selected_indices.append(remaining_indices[0])
        remaining_indices.remove(remaining_indices[0])
        
        # Select remaining items using MMR
        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []
            
            for i in remaining_indices:
                # Relevance score (similarity to query)
                relevance = case_scores[i]
                
                # Diversity score (negative maximum similarity to already selected items)
                if selected_indices:
                    selected_embeddings = case_embeddings[selected_indices]
                    similarity_to_selected = np.max(
                        np.dot(case_embeddings[i], selected_embeddings.T)
                    )
                else:
                    similarity_to_selected = 0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * similarity_to_selected
                mmr_scores.append((i, mmr_score))
            
            # Select item with highest MMR score
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            next_idx = mmr_scores[0][0]
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
        
        # Return selected case IDs and original scores
        return [(case_ids[idx], case_scores[idx]) for idx in selected_indices]
    
    def search_cases(self, query: str, k: int = None, filter_case_ids: List[int] = None, include_highlights: bool = False) -> List[SearchResult]:
        """
        Search for cases similar to the query.
        
        Args:
            query: Query string
            k: Number of results to return (defaults to config["top_final"])
            filter_case_ids: Optional list of case IDs to filter results
            include_highlights: Whether to include highlights in the results (default: False)
            
        Returns:
            List of SearchResult objects
        """
        if k is None:
            k = self.config["top_final"]
        
        start_time = time.time()
        logger.info(f"Searching for cases similar to: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        logger.info(f"Generated query embedding with shape {query_embedding.shape}")
        
        # If filter_case_ids is provided, only return those specific cases
        if filter_case_ids:
            logger.info(f"Filtering results to case IDs: {filter_case_ids}")
            
            # One global sentence search if highlights are needed
            highlights_by_case = {}
            if include_highlights:
                global_sent_hits = self.response_index.search(query_embedding[0], 500)
                
                # Group hits by case_id
                hits_by_case: Dict[int, List[Hit]] = {}
                for h in global_sent_hits:
                    cid = h.payload.get("case_id")
                    if cid is not None:
                        hits_by_case.setdefault(cid, []).append(h)
                
                # Process highlights for each case
                for case_id in filter_case_ids:
                    try:
                        case = next(c for c in self.cases if c.id == case_id)
                        resp_sents = case.response_sentences or []
                        
                        # Take top M hits for this case
                        case_hits = hits_by_case.get(case_id, [])
                        
                        # Normalize per-case by its max score to [0,1]
                        if case_hits:
                            mx = max(h.score for h in case_hits if h.score is not None)
                        else:
                            mx = 0.0
                            
                        case_highlights = []
                        for h in sorted(case_hits, key=lambda x: x.score, reverse=True)[:self.config["m"]]:
                            sid = h.payload["sent_id"]
                            if 0 <= sid < len(resp_sents) and mx > 0:
                                s = resp_sents[sid]
                                case_highlights.append(Highlight(
                                    sent_id=sid, text=s.text, start=s.start, end=s.end,
                                    score=float(h.score / mx)
                                ))
                        highlights_by_case[case_id] = case_highlights
                    except Exception as e:
                        logger.error(f"Error processing highlights for case {case_id}: {e}")
            
            search_results = []
            for case_id in filter_case_ids:
                try:
                    case = next(c for c in self.cases if c.id == case_id)
                    
                    # For filtered cases, use a default cosine score
                    search_results.append(SearchResult(
                        case_id=case_id,
                        context=case.context,
                        response=case.response,
                        score=0.5,  # Default score when filtering by ID
                        highlights=highlights_by_case.get(case_id, [])
                    ))
                except StopIteration:
                    logger.warning(f"Case not found: {case_id}")
                except Exception as e:
                    logger.error(f"Error processing case {case_id}: {e}")
            
            logger.info(f"Found {len(search_results)} filtered cases in {time.time() - start_time:.2f}s")
            return search_results
        
        # Step 1: Dense retrieval from context index
        logger.info(f"Performing dense search with k={self.config['k1']}")
        dense_hits = self.context_index.search(query_embedding[0], self.config["k1"])
        logger.info(f"Dense search returned {len(dense_hits)} hits")
        
        # Store dense cosine scores for display and MMR
        dense_dict = {h.payload["case_id"]: float(h.score) for h in dense_hits}  # cosine
        dense_results = list(dense_dict.items())  # [(case_id, cosine)]
        logger.info(f"Dense results: {dense_results[:3]}")
        
        # Step 2: Lexical search using BM25
        logger.info(f"Performing lexical search with k={self.config['k1']}")
        lexical_results = self._lexical_search(query, self.config["k1"])
        logger.info(f"Lexical results: {lexical_results[:3]}")
        
        # Step 3: RRF fusion
        logger.info("Performing RRF fusion")
        fused_results = self._rrf_fusion(dense_results, lexical_results)  # [(case_id, rrf_score)]
        logger.info(f"Fused results: {fused_results[:3]}")
        
        # Keep top-N by RRF for probing
        top_n = fused_results[:self.config["n"]]
        top_case_ids = [cid for cid, _ in top_n]
        
        # Check if top score is below threshold
        if top_n and top_n[0][1] < self.config["min_score_threshold"]:
            logger.warning(f"Top case score {top_n[0][1]} is below threshold {self.config['min_score_threshold']}")
            return []
        
        # Build aligned lists for MMR using dense cosine as relevance
        case_scores_for_mmr = [dense_dict.get(cid, 0.0) for cid in top_case_ids]
        
        # Step 4: MMR diversification
        logger.info(f"Performing MMR diversification with k={min(k, len(top_case_ids))}")
        diversified = self._mmr_diversification(
            top_case_ids, 
            case_scores_for_mmr, 
            query_embedding[0], 
            min(k, len(top_case_ids))
        )
        logger.info(f"Diversified results: {diversified}")
        
        # Step 5: Process sentence highlights if needed
        highlights_by_case = {}
        if include_highlights:
            # One global sentence search
            global_sent_hits = self.response_index.search(query_embedding[0], 500)
            
            # Group hits by case_id
            hits_by_case: Dict[int, List[Hit]] = {}
            for h in global_sent_hits:
                cid = h.payload.get("case_id")
                if cid is not None:
                    hits_by_case.setdefault(cid, []).append(h)
            
            # Process highlights for each case
            for case_id, _ in diversified:
                # Fetch case
                case = next((c for c in self.cases if c.id == case_id), None)
                if not case:
                    continue
                
                resp_sents = case.response_sentences or []
                
                # Take top M hits for this case
                case_hits = hits_by_case.get(case_id, [])
                
                # Normalize per-case by its max score to [0,1]
                if case_hits:
                    mx = max(h.score for h in case_hits if h.score is not None)
                else:
                    mx = 0.0
                    
                case_highlights = []
                for h in sorted(case_hits, key=lambda x: x.score, reverse=True)[:self.config["m"]]:
                    sid = h.payload["sent_id"]
                    if 0 <= sid < len(resp_sents) and mx > 0:
                        s = resp_sents[sid]
                        case_highlights.append(Highlight(
                            sent_id=sid, text=s.text, start=s.start, end=s.end,
                            score=float(h.score / mx)
                        ))
                highlights_by_case[case_id] = case_highlights
        
        search_results = []
        for case_id, case_score in diversified:
            # Fetch case
            case = next((c for c in self.cases if c.id == case_id), None)
            if not case:
                continue
            
            # Visible "score" = dense cosine
            visible_score = dense_dict.get(case_id, 0.0)
            search_results.append(SearchResult(
                case_id=case_id, 
                context=case.context, 
                response=case.response,
                score=visible_score, 
                highlights=highlights_by_case.get(case_id, [])
            ))
        
        logger.info(f"Found {len(search_results)} cases in {time.time() - start_time:.2f}s")
        return search_results 