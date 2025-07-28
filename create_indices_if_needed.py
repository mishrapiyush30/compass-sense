#!/usr/bin/env python3
import os
import sys
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentenceSpan:
    """Represents a sentence with its character offsets in the original text."""
    def __init__(self, text: str, start: int, end: int):
        self.text = text
        self.start = start
        self.end = end

class CaseItem:
    """Represents a case from the counseling Q&A corpus."""
    def __init__(self, id: int, context: str, response: str, response_sentences: List[SentenceSpan] = None):
        self.id = id
        self.context = context
        self.response = response
        self.response_sentences = response_sentences or []

def load_ndjson(file_path: str):
    """Load NDJSON file line by line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON on line {i+1}, skipping")

def split_sentences(text: str) -> List[SentenceSpan]:
    """Split text into sentences with character offsets."""
    # Simple sentence splitting by periods, question marks, and exclamation marks
    sentence_ends = []
    i = 0
    while i < len(text):
        if text[i] in ['.', '!', '?']:
            sentence_ends.append(i)
            i += 1
            # Skip any following punctuation
            while i < len(text) and text[i] in ['.', '!', '?', ' ', '\n']:
                i += 1
        else:
            i += 1
    
    # Add the end of the text
    if not sentence_ends or sentence_ends[-1] < len(text) - 1:
        sentence_ends.append(len(text) - 1)
    
    # Create sentence spans
    spans = []
    start = 0
    for end in sentence_ends:
        # Add 1 to include the punctuation
        end_pos = min(end + 1, len(text))
        sentence = text[start:end_pos].strip()
        if sentence:  # Skip empty sentences
            spans.append(SentenceSpan(text=sentence, start=start, end=end_pos))
        start = end_pos
    
    return spans

def load_dataset(file_path: str, max_items: Optional[int] = None) -> List[CaseItem]:
    """Load and preprocess the dataset."""
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
            pair_hash = hash(f"{context}:{response}")
            
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

def check_indices_exist(index_dir: str) -> bool:
    """Check if indices already exist."""
    required_files = [
        "ctx.index", 
        "resp.index", 
        "ctx.payloads.json", 
        "resp.payloads.json", 
        "index_manifest.json"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(index_dir, file)):
            logger.info(f"Missing required file: {file}")
            return False
    
    logger.info("All required index files exist")
    return True

def create_indices(cases: List[CaseItem], model_name: str, output_dir: str):
    """Create and save indices."""
    logger.info(f"Creating indices using model: {model_name}")
    start_time = time.time()
    
    # Load model
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded. Vector dimension: {dim}")
    
    # Create context index
    logger.info("Building context index (Index A)...")
    contexts = [case.context for case in cases]
    context_embeddings = model.encode(contexts, normalize_embeddings=True, show_progress_bar=True)
    
    context_index = faiss.IndexFlatIP(dim)
    context_index.add(context_embeddings.astype(np.float32))
    
    # Create context payloads
    context_payloads = []
    for i, case in enumerate(cases):
        context_payloads.append({
            "case_id": case.id,
            "length": len(case.context),
            "hash": hash(case.context) % 10000000
        })
    
    # Create response sentence index
    logger.info("Building response sentence index (Index B)...")
    all_sentences = []
    sentence_payloads = []
    
    for case_idx, case in enumerate(cases):
        for sent_idx, sentence in enumerate(case.response_sentences):
            all_sentences.append(sentence.text)
            sentence_payloads.append({
                "case_id": case.id,
                "sent_id": sent_idx,
                "start": sentence.start,
                "end": sentence.end,
                "text": sentence.text,
                "score": 0.0
            })
    
    sentence_embeddings = model.encode(all_sentences, normalize_embeddings=True, show_progress_bar=True)
    
    response_index = faiss.IndexFlatIP(dim)
    response_index.add(sentence_embeddings.astype(np.float32))
    
    # Save indices
    logger.info("Saving indices...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save context index
    faiss.write_index(context_index, os.path.join(output_dir, "ctx.index"))
    with open(os.path.join(output_dir, "ctx.payloads.json"), "w") as f:
        json.dump(context_payloads, f)
    
    # Save response index
    faiss.write_index(response_index, os.path.join(output_dir, "resp.index"))
    with open(os.path.join(output_dir, "resp.payloads.json"), "w") as f:
        json.dump(sentence_payloads, f)
    
    # Save manifest
    manifest = {
        "version": "1.0.0",
        "created_at": time.time(),
        "embed_model": model_name,
        "vector_dim": dim,
        "context_count": len(context_payloads),
        "sentence_count": len(sentence_payloads),
        "case_count": len(cases),
        "checksum": hash(str(context_payloads) + str(sentence_payloads)) % 1000000
    }
    
    with open(os.path.join(output_dir, "index_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Save cases
    with open(os.path.join(os.path.dirname(output_dir), "cases.json"), "w") as f:
        cases_json = []
        for case in cases:
            case_dict = {
                "id": case.id,
                "context": case.context,
                "response": case.response,
                "response_sentences": [
                    {"text": s.text, "start": s.start, "end": s.end}
                    for s in case.response_sentences
                ]
            }
            cases_json.append(case_dict)
        json.dump(cases_json, f)
    
    logger.info(f"Indices built and saved in {output_dir} in {time.time() - start_time:.2f}s")
    
    return context_index, response_index, context_payloads, sentence_payloads

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create indices if they don't exist")
    parser.add_argument("--data-file", default="data/combined_dataset.json", help="Path to the dataset file")
    parser.add_argument("--output-dir", default="data/indices", help="Directory to save indices")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model to use")
    parser.add_argument("--max-items", type=int, help="Maximum number of items to process")
    parser.add_argument("--force", action="store_true", help="Force recreation of indices even if they exist")
    args = parser.parse_args()
    
    # Check if indices already exist
    if not args.force and check_indices_exist(args.output_dir):
        logger.info("Indices already exist. Use --force to recreate them.")
        return
    
    # Load dataset
    cases = load_dataset(args.data_file, args.max_items)
    
    # Create indices
    create_indices(cases, args.model, args.output_dir)

if __name__ == "__main__":
    main() 