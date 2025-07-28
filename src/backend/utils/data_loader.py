import json
import logging
from typing import List, Dict, Any, Optional, Iterator
import nltk
from nltk.tokenize import sent_tokenize
import os
import hashlib

from ..models.schema import CaseItem, SentenceSpan

logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer")
    nltk.download('punkt', quiet=True)

# Also download punkt_tab for newer NLTK versions
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("Downloading NLTK punkt_tab tokenizer")
    nltk.download('punkt_tab', quiet=True)


def normalize_text(text: str) -> str:
    """Normalize whitespace but preserve punctuation and casing."""
    # Remove extra whitespace but keep line breaks as single spaces
    return " ".join(text.split())


def split_sentences(text: str) -> List[SentenceSpan]:
    """
    Split text into sentences with character offsets.
    
    Args:
        text: Text to split into sentences
        
    Returns:
        List of SentenceSpan objects with text and character offsets
    """
    # First get sentence boundaries
    sentences = sent_tokenize(text)
    
    # Now find the character offsets
    spans = []
    start = 0
    for sentence in sentences:
        # Find the sentence in the original text
        # We need to do this to get the correct offsets with original whitespace
        sentence_stripped = normalize_text(sentence)
        found_start = text.find(sentence, start)
        
        # If not found with exact match, try with normalized whitespace
        if found_start == -1:
            # This is a fallback if the exact sentence isn't found
            # It can happen due to whitespace differences
            logger.warning(f"Sentence not found exactly, using approximate match: {sentence_stripped}")
            start_candidates = []
            words = sentence_stripped.split()
            if words:
                # Try to find the first few words
                search_start = " ".join(words[:3]) if len(words) >= 3 else words[0]
                found_idx = text.find(search_start, start)
                if found_idx != -1:
                    start_candidates.append(found_idx)
            
            # If we found candidates, use the first one
            if start_candidates:
                found_start = min(start_candidates)
            else:
                # If still not found, just append to the end with a note
                logger.warning(f"Could not find sentence in text, appending to end: {sentence_stripped}")
                found_start = len(text)
        
        found_end = found_start + len(sentence)
        spans.append(SentenceSpan(text=sentence, start=found_start, end=found_end))
        start = found_end
    
    return spans


def load_ndjson(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    Load NDJSON file line by line.
    
    Args:
        file_path: Path to NDJSON file
        
    Yields:
        Dictionary for each line
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON on line {i+1}, skipping")


def load_dataset(file_path: str, max_items: Optional[int] = None) -> List[CaseItem]:
    """
    Load and preprocess the dataset.
    
    Args:
        file_path: Path to NDJSON file
        max_items: Maximum number of items to load (for testing)
        
    Returns:
        List of CaseItem objects
    """
    logger.info(f"Loading dataset from {file_path}")
    
    cases = []
    seen_hashes = set()  # For deduplication
    
    for i, record in enumerate(load_ndjson(file_path)):
        if max_items and i >= max_items:
            break
            
        try:
            # Extract and normalize fields
            context = record.get("Context", "").strip()
            response = record.get("Response", "").strip()
            
            if not context or not response:
                logger.warning(f"Skipping record {i} with empty context or response")
                continue
            
            # Create hash for deduplication
            pair_hash = hashlib.md5(f"{context}:{response}".encode()).hexdigest()
            
            # Skip if exact duplicate
            if pair_hash in seen_hashes:
                logger.debug(f"Skipping duplicate record {i}")
                continue
            
            seen_hashes.add(pair_hash)
            
            # Split response into sentences with offsets
            response_sentences = split_sentences(response)
            
            # Create case item
            case = CaseItem(
                id=i,
                context=context,
                response=response,
                response_sentences=response_sentences
            )
            
            cases.append(case)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i+1} records, kept {len(cases)}")
                
        except Exception as e:
            logger.error(f"Error processing record {i}: {e}")
    
    logger.info(f"Loaded {len(cases)} unique cases from {file_path}")
    return cases


def save_cases(cases: List[CaseItem], output_path: str) -> None:
    """
    Save cases to a JSON file.
    
    Args:
        cases: List of CaseItem objects
        output_path: Path to save the cases
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert cases to dictionaries
    cases_dict = []
    for case in cases:
        case_dict = case.model_dump()
        cases_dict.append(case_dict)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cases_dict, f, indent=2)
    
    logger.info(f"Saved {len(cases)} cases to {output_path}")


def load_saved_cases(input_path: str) -> List[CaseItem]:
    """
    Load cases from a JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        List of CaseItem objects
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        cases_dict = json.load(f)
    
    cases = []
    for case_dict in cases_dict:
        # Create CaseItem
        case = CaseItem(**case_dict)
        cases.append(case)
    
    logger.info(f"Loaded {len(cases)} cases from {input_path}")
    return cases 