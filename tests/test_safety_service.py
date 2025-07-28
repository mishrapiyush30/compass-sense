import unittest
from src.backend.services.safety_service import SafetyService

class TestSafetyService(unittest.TestCase):
    """Tests for the SafetyService."""
    
    def setUp(self):
        """Set up the test case."""
        self.safety_service = SafetyService()
    
    def test_detect_crisis(self):
        """Test crisis detection."""
        # Test positive cases
        self.assertTrue(self.safety_service.detect_crisis("I feel suicidal"))
        self.assertTrue(self.safety_service.detect_crisis("I want to kill myself"))
        self.assertTrue(self.safety_service.detect_crisis("I shouldn't be here anymore"))
        self.assertTrue(self.safety_service.detect_crisis("I want to end it all"))
        
        # Test negative cases
        self.assertFalse(self.safety_service.detect_crisis("I feel anxious"))
        self.assertFalse(self.safety_service.detect_crisis("I'm having a hard time"))
        self.assertFalse(self.safety_service.detect_crisis("I need help with my anxiety"))
    
    def test_filter_evidence(self):
        """Test evidence filtering."""
        # Test positive cases (safe evidence)
        self.assertTrue(self.safety_service.filter_evidence("It's normal to feel anxious sometimes"))
        self.assertTrue(self.safety_service.filter_evidence("Taking a walk can help clear your mind"))
        
        # Test negative cases (unsafe evidence)
        self.assertFalse(self.safety_service.filter_evidence("You should take medication for that"))
        self.assertFalse(self.safety_service.filter_evidence("This will definitely cure your anxiety"))
        self.assertFalse(self.safety_service.filter_evidence("Alcohol might help you relax"))
    
    def test_compute_rouge_l(self):
        """Test ROUGE-L computation."""
        # Test exact match
        self.assertEqual(
            self.safety_service.compute_rouge_l("This is a test", "This is a test"),
            1.0
        )
        
        # Test partial match
        score = self.safety_service.compute_rouge_l(
            "This is a test sentence",
            "This is a different sentence"
        )
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
        
        # Test no match
        score = self.safety_service.compute_rouge_l(
            "This is a test",
            "Something completely different"
        )
        self.assertLess(score, 0.5)
    
    def test_verify_evidence(self):
        """Test evidence verification."""
        # Test sufficient evidence
        is_valid, score = self.safety_service.verify_evidence(
            "It's normal to feel anxious sometimes",
            ["It's normal to feel anxious sometimes", "Many people experience anxiety"]
        )
        self.assertTrue(is_valid)
        self.assertGreaterEqual(score, self.safety_service.config["gate_alpha"])
        
        # Test insufficient evidence
        is_valid, score = self.safety_service.verify_evidence(
            "You should take medication for your anxiety",
            ["It's normal to feel anxious sometimes", "Many people experience anxiety"]
        )
        self.assertFalse(is_valid)
        self.assertLess(score, self.safety_service.config["gate_alpha"])
    
    def test_run_evidence_gate(self):
        """Test evidence gate."""
        # Test gate passes
        sections = {
            "validation": "It's normal to feel anxious sometimes",
            "reflection": "I hear that you're feeling overwhelmed by your anxiety"
        }
        quotes = [
            "It's normal to feel anxious sometimes",
            "Many people feel overwhelmed by anxiety"
        ]
        
        valid_sections, section_scores, gate_passed = self.safety_service.run_evidence_gate(
            sections, quotes
        )
        
        self.assertTrue(gate_passed)
        self.assertIn("validation", valid_sections)
        self.assertIn("reflection", valid_sections)
        
        # Test gate fails
        sections = {
            "validation": "You should take medication for your anxiety",
            "reflection": "I hear that you're feeling overwhelmed by your anxiety"
        }
        
        valid_sections, section_scores, gate_passed = self.safety_service.run_evidence_gate(
            sections, quotes
        )
        
        self.assertFalse(gate_passed)
        self.assertNotIn("validation", valid_sections)
        self.assertIn("reflection", valid_sections)


if __name__ == "__main__":
    unittest.main() 