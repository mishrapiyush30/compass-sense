import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
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

def _sentences(text: str) -> List[str]:
    """Robust sentence splitter with NLTK fallback to regex."""
    try:
        from nltk.tokenize import sent_tokenize
        sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        sents = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    return sents


class SafetyService:
    """Service for safety checks and evidence verification."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the safety service.
        
        Args:
            config: Configuration parameters
        """
        # Default configuration (do not mutate during requests)
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
        Legacy boolean check (kept for backward compatibility).
        
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

    def crisis_severity(self, text: str) -> str:
        """
        Determine crisis severity with simple, explainable rules.
        Returns: 'none' | 'mild' | 'moderate' | 'high'
        """
        t = text.lower()
        # Negation guard: "I'm not suicidal"
        if re.search(r"\b(not|no|never)\b.{0,30}\b(suicid(e|al)|kill myself|end my life|want to die)\b", t):
            return "mild"
        direct = re.search(r"\b(suicid(e|al)|kill myself|end my life|take my (own )?life|want to die)\b", t)
        plan   = re.search(r"\b(plan|method|means|bought (rope|pills)|set a date)\b", t)
        timeb  = re.search(r"\b(today|tonight|right now|soon|this week)\b", t)
        harm   = re.search(r"\b(hurt myself|harm myself|hurt (someone|others))\b", t)
        if direct and (plan or timeb):
            return "high"
        if direct or harm:
            return "moderate"
        return "none"
    
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

    def verify_sentence(
        self,
        sentence: str,
        quotes: List[str],
        alpha: float = 0.25,
        topk: int = 5
    ) -> Tuple[bool, float, List[str]]:
        """
        Verify a single sentence against a set of quotes using ROUGE-L.
        Returns (supported?, best_score, top_quotes_used)
        """
        if not quotes:
            return False, 0.0, []
        scored = [(self.compute_rouge_l(sentence, q), q) for q in quotes]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:topk]
        best = top[0][0] if top else 0.0
        return best >= alpha, best, [q for _, q in top]
    
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

    def run_evidence_gate(
        self,
        sections: Dict[str, str],
        quotes: List[str],
        *,
        alpha: float = 0.25,
        min_supported_sentences: int = 2,
        require_distinct_cases: int = 2,
        quote_to_case: Optional[Dict[str, int]] = None
    ) -> Tuple[Dict[str, str], Dict[str, float], bool]:
        """
        Sentence-level gate with case diversity.
        
        Args:
            sections: {section_name: text}
            quotes: list of quote strings (sentences)
            alpha: per-sentence ROUGE-L threshold
            min_supported_sentences: min sentences per section that must be supported
            require_distinct_cases: min distinct case_ids cited
            quote_to_case: optional mapping quote_text -> case_id
            
        Returns:
            Tuple of (valid_sections, section_scores, gate_passed)
        """
        valid_sections: Dict[str, str] = {}
        section_scores: Dict[str, float] = {}
        quote_to_case = quote_to_case or {}

        for name, text in sections.items():
            if not text:
                continue
            sents = _sentences(text)
            if not sents:
                continue

            supported = 0
            per_sent_scores: List[float] = []
            used_cases: Set[int] = set()
            used_quotes: List[str] = []

            for s in sents:
                ok, sc, top_qs = self.verify_sentence(s, quotes, alpha=alpha)
                per_sent_scores.append(sc)
                if ok:
                    supported += 1
                    used_quotes.extend(top_qs)
                    for q in top_qs:
                        cid = quote_to_case.get(q)
                        if cid is not None:
                            used_cases.add(cid)

            per_sent_scores.sort()
            agg = per_sent_scores[len(per_sent_scores)//2] if per_sent_scores else 0.0
            section_scores[name] = float(agg)

            if supported >= min_supported_sentences and len(used_cases) >= require_distinct_cases:
                valid_sections[name] = text

        gate_passed = len(valid_sections) > 0
        logger.info(f"Evidence gate: passed={gate_passed}, scores={section_scores}")
        return valid_sections, section_scores, gate_passed

    def contains_denied_terms(self, text: str) -> bool:
        """Scan final generated text for deny-listed terms."""
        return any(p.search(text) for p in self.evidence_denylist) 