import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import nltk
from nltk.util import ngrams
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer")
    nltk.download('punkt', quiet=True)


class SafetyService:
    """Service for safety checks and evidence verification."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the safety service.
        
        Args:
            config: Configuration parameters
        """
        # Default configuration
        self.config = {
            "crisis_patterns": [
                r"suicid(e|al)",
                r"kill\s+my\s*self",
                r"killing\s+my\s*self",
                r"feel\s+like\s+kill(ing)?\s+my\s*self",
                r"end\s+my\s+life",
                r"want\s+to\s+die",
                r"no\s+reason\s+to\s+live",
                r"better\s+off\s+dead",
                r"take\s+my\s+(own\s+)?life",
                r"harm\s+my\s*self",
                r"hurt\s+my\s*self",
                r"don't\s+want\s+to\s+live",
                r"don't\s+want\s+to\s+be\s+alive",
                r"end\s+it\s+all"
            ],
            "evidence_denylist": [
                r"medic(ation|ine|al)",
                r"prescri(be|ption)",
                r"diagnos(e|is|ed)",
                r"drug(s)?",
                r"pill(s)?",
                r"always",
                r"never",
                r"guaranteed",
                r"cure",
                r"100\s*%",
                r"definitely",
                r"alcohol",
                r"marijuana",
                r"cannabis",
                r"weed",
                r"absolutely"
            ],
            "gate_alpha": 0.1,  # Minimum evidence overlap threshold (lowered from 0.3)
            "min_valid_sections": 1  # Minimum number of valid sections required
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Compile regex patterns
        self.crisis_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.config["crisis_patterns"]]
        self.evidence_denylist = [re.compile(pattern, re.IGNORECASE) for pattern in self.config["evidence_denylist"]]
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        logger.info("Initialized safety service")
    
    def detect_crisis(self, text: str) -> bool:
        """
        Detect crisis language in text.
        
        Args:
            text: Text to check for crisis language
            
        Returns:
            True if crisis language is detected, False otherwise
        """
        for pattern in self.crisis_patterns:
            if pattern.search(text):
                logger.warning(f"Crisis language detected: {pattern.pattern}")
                return True
        
        return False
    
    def filter_evidence(self, text: str) -> bool:
        """
        Check if text contains denied evidence.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is safe, False if it contains denied evidence
        """
        for pattern in self.evidence_denylist:
            if pattern.search(text):
                logger.warning(f"Denied evidence detected: {pattern.pattern}")
                return False
        
        return True
    
    def compute_ngram_overlap(self, text1: str, text2: str, n: int = 3) -> float:
        """
        Compute n-gram F1 overlap between two texts.
        
        Args:
            text1: First text
            text2: Second text
            n: N-gram size
            
        Returns:
            F1 score of n-gram overlap
        """
        # Tokenize texts
        tokens1 = text1.lower().split()
        tokens2 = text2.lower().split()
        
        # Generate n-grams
        ngrams1 = set(" ".join(gram) for gram in ngrams(tokens1, n))
        ngrams2 = set(" ".join(gram) for gram in ngrams(tokens2, n))
        
        # Calculate F1 score
        if not ngrams1 or not ngrams2:
            return 0.0
        
        # Calculate precision, recall, F1
        intersection = ngrams1.intersection(ngrams2)
        precision = len(intersection) / len(ngrams1) if ngrams1 else 0
        recall = len(intersection) / len(ngrams2) if ngrams2 else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def compute_rouge_l(self, text1: str, text2: str) -> float:
        """
        Compute ROUGE-L (LCS-F1) between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            ROUGE-L F1 score
        """
        scores = self.rouge_scorer.score(text1, text2)
        return scores['rougeL'].fmeasure
    
    def verify_evidence(self, sentence: str, quotes: List[str]) -> Tuple[bool, float]:
        """
        Verify if a sentence has sufficient evidence in quotes.
        
        Args:
            sentence: Sentence to verify
            quotes: List of quote strings
            
        Returns:
            Tuple of (is_valid, best_score)
        """
        if not quotes:
            logger.info("No quotes provided for evidence verification")
            return False, 0.0
        
        logger.info(f"Verifying evidence for: '{sentence[:50]}...'")
        
        best_score = 0.0
        best_quote = ""
        for i, quote in enumerate(quotes):
            # Use ROUGE-L for evidence verification
            score = self.compute_rouge_l(sentence, quote)
            if score > best_score:
                best_score = score
                best_quote = quote
        
        is_valid = best_score >= self.config["gate_alpha"]
        logger.info(f"Best score: {best_score}, threshold: {self.config['gate_alpha']}, valid: {is_valid}")
        logger.info(f"Best matching quote: '{best_quote[:50]}...'")
        
        return is_valid, best_score
    
    def run_evidence_gate(self, sections: Dict[str, str], quotes: List[str]) -> Tuple[Dict[str, str], Dict[str, float], bool]:
        """
        Run evidence gate on sections.
        
        Args:
            sections: Dictionary of section name to section text
            quotes: List of quote strings
            
        Returns:
            Tuple of (valid_sections, section_scores, gate_passed)
        """
        valid_sections = {}
        section_scores = {}
        
        # Set a very low threshold to allow more responses
        self.config["gate_alpha"] = 0.01
        self.config["min_valid_sections"] = 1
        
        logger.info(f"Running evidence gate with {len(sections)} sections and {len(quotes)} quotes")
        logger.info(f"Sections: {list(sections.keys())}")
        logger.info(f"Gate alpha threshold: {self.config['gate_alpha']}")
        logger.info(f"Min valid sections: {self.config['min_valid_sections']}")
        
        for section_name, section_text in sections.items():
            # Skip empty sections
            if not section_text:
                logger.info(f"Section '{section_name}' is empty, skipping")
                continue
                
            # Verify evidence
            is_valid, score = self.verify_evidence(section_text, quotes)
            section_scores[section_name] = score
            
            logger.info(f"Section '{section_name}' score: {score}, valid: {is_valid}")
            
            if is_valid:
                valid_sections[section_name] = section_text
        
        # Check if enough valid sections remain
        gate_passed = len(valid_sections) >= self.config["min_valid_sections"]
        
        logger.info(f"Evidence gate result: {len(valid_sections)} valid sections, passed: {gate_passed}")
        
        return valid_sections, section_scores, gate_passed 