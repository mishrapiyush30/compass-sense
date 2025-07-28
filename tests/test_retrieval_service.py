import unittest
import numpy as np
from unittest.mock import Mock, patch

from src.backend.services.retrieval_service import RetrievalService
from src.backend.models.schema import CaseItem, SentenceSpan


class TestRetrievalService(unittest.TestCase):
    """Tests for the RetrievalService."""
    
    def setUp(self):
        """Set up the test case."""
        # Create mock cases
        self.cases = [
            CaseItem(
                id=0,
                context="I feel anxious and overwhelmed with work",
                response="It's normal to feel anxious sometimes. Taking breaks can help.",
                response_sentences=[
                    SentenceSpan(text="It's normal to feel anxious sometimes.", start=0, end=36),
                    SentenceSpan(text="Taking breaks can help.", start=37, end=59)
                ]
            ),
            CaseItem(
                id=1,
                context="I'm feeling depressed and don't know what to do",
                response="Depression is common. Talking to someone can be helpful.",
                response_sentences=[
                    SentenceSpan(text="Depression is common.", start=0, end=19),
                    SentenceSpan(text="Talking to someone can be helpful.", start=20, end=54)
                ]
            ),
            CaseItem(
                id=2,
                context="I'm having trouble sleeping at night",
                response="Sleep issues can be caused by stress. Try a bedtime routine.",
                response_sentences=[
                    SentenceSpan(text="Sleep issues can be caused by stress.", start=0, end=36),
                    SentenceSpan(text="Try a bedtime routine.", start=37, end=59)
                ]
            )
        ]
        
        # Create mock vector stores
        self.context_index = Mock()
        self.response_index = Mock()
        
        # Create mock embedding service
        self.embedding_service = Mock()
        self.embedding_service.embed_text.return_value = np.ones((1, 384))
        
        # Create retrieval service
        self.retrieval_service = RetrievalService(
            context_index=self.context_index,
            response_index=self.response_index,
            embedding_service=self.embedding_service,
            cases=self.cases
        )
    
    def test_lexical_search(self):
        """Test lexical search."""
        # Mock BM25
        self.retrieval_service.bm25 = Mock()
        self.retrieval_service.bm25.get_scores.return_value = np.array([0.8, 0.5, 0.2])
        
        # Test lexical search
        results = self.retrieval_service._lexical_search("anxiety", 3)
        
        # Check results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 0)  # First case ID
        self.assertEqual(results[1][0], 1)  # Second case ID
        self.assertEqual(results[2][0], 2)  # Third case ID
        
        # Check scores are normalized
        self.assertEqual(results[0][1], 1.0)  # First score (normalized)
        self.assertLess(results[1][1], 1.0)  # Second score (normalized)
        self.assertLess(results[2][1], 1.0)  # Third score (normalized)
    
    def test_rrf_fusion(self):
        """Test RRF fusion."""
        # Test inputs
        dense_results = [(0, 0.9), (1, 0.7), (2, 0.5)]
        lexical_results = [(0, 0.8), (2, 0.6), (1, 0.4)]
        
        # Test RRF fusion
        fused_results = self.retrieval_service._rrf_fusion(dense_results, lexical_results)
        
        # Check results
        self.assertEqual(len(fused_results), 3)
        
        # First result should be case 0 (best in both)
        self.assertEqual(fused_results[0][0], 0)
        
        # Scores should be RRF scores
        for case_id, score in fused_results:
            self.assertGreater(score, 0.0)
    
    def test_mmr_diversification(self):
        """Test MMR diversification."""
        # Mock embedding service to return different embeddings for each case
        embeddings = [
            np.array([1.0, 0.0, 0.0]),  # Case 0
            np.array([0.0, 1.0, 0.0]),  # Case 1
            np.array([0.0, 0.0, 1.0])   # Case 2
        ]
        
        def mock_embed_text(text):
            if "anxious" in text:
                return embeddings[0].reshape(1, -1)
            elif "depressed" in text:
                return embeddings[1].reshape(1, -1)
            else:
                return embeddings[2].reshape(1, -1)
        
        self.embedding_service.embed_text.side_effect = mock_embed_text
        
        # Test inputs
        case_ids = [0, 1, 2]
        case_scores = [0.9, 0.8, 0.7]
        query_embedding = np.array([0.5, 0.5, 0.0])  # More similar to cases 0 and 1
        
        # Test MMR diversification
        results = self.retrieval_service._mmr_diversification(
            case_ids, case_scores, query_embedding, 2
        )
        
        # Check results
        self.assertEqual(len(results), 2)
        
        # First result should be case 0 (highest score)
        self.assertEqual(results[0][0], 0)
        
        # Second result should be case 1 or 2 (diverse from case 0)
        self.assertIn(results[1][0], [1, 2])
    
    @patch('src.backend.services.retrieval_service.RetrievalService._lexical_search')
    @patch('src.backend.services.retrieval_service.RetrievalService._rrf_fusion')
    @patch('src.backend.services.retrieval_service.RetrievalService._mmr_diversification')
    def test_search_cases(self, mock_mmr, mock_rrf, mock_lexical):
        """Test search_cases method."""
        # Mock context index search
        self.context_index.search.return_value = [
            Mock(payload={"case_id": 0}, score=0.9),
            Mock(payload={"case_id": 1}, score=0.7),
            Mock(payload={"case_id": 2}, score=0.5)
        ]
        
        # Mock lexical search
        mock_lexical.return_value = [(0, 0.8), (2, 0.6), (1, 0.4)]
        
        # Mock RRF fusion
        mock_rrf.return_value = [(0, 0.9), (1, 0.7), (2, 0.5)]
        
        # Mock MMR diversification
        mock_mmr.return_value = [(0, 0.9), (2, 0.5)]
        
        # Mock response index search
        self.response_index.search.return_value = [
            Mock(payload={"case_id": 0, "sent_id": 0, "start": 0, "end": 36, "text": "It's normal to feel anxious sometimes.", "score": 0.8}),
            Mock(payload={"case_id": 0, "sent_id": 1, "start": 37, "end": 59, "text": "Taking breaks can help.", "score": 0.7})
        ]
        
        # Test search_cases
        results = self.retrieval_service.search_cases("anxiety", 2)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].case_id, 0)
        self.assertEqual(results[1].case_id, 2)
        
        # Check highlights
        self.assertEqual(len(results[0].highlights), 2)
        self.assertEqual(results[0].highlights[0].text, "It's normal to feel anxious sometimes.")
        
        # Check that all methods were called
        self.embedding_service.embed_text.assert_called()
        self.context_index.search.assert_called()
        mock_lexical.assert_called()
        mock_rrf.assert_called()
        mock_mmr.assert_called()
        self.response_index.search.assert_called()


if __name__ == "__main__":
    unittest.main() 