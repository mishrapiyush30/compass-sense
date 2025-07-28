# CoachCritique

Safety-First Coaching with Bounded LLM & Evidence Gate

## Overview

CoachCritique is a safety-first coaching system that:

1. Given a user concern (e.g., "I'm feeling depressed and anxious"), retrieves the **3 most similar cases** from a counseling Q&A corpus.
2. On request, produces a **safety-first, evidence-cited coaching reply**.
3. Enforces strict safety with a **bounded LLM** that only selects **skills & quotes in JSON** (never writes free text).
4. Assembles the final reply via **templates** and runs an **Evidence Gate** to ensure safety.
5. Uses hybrid retrieval (dense + lexical) with **observable** results.

## Features

- **Top-3 Similar Cases**: Returns the most relevant cases with highlighted, quotable evidence.
- **Structured Coaching Reply**: Produces a structured reply (Validation, Reflection, One small step, Goals, Resources) with citations.
- **Safety Gate**: Crisis language triggers resources-only mode; no coaching.
- **Evidence Gate**: Ensures all coaching advice has sufficient evidence (overlap ≥ 0.6) from the corpus.
- **Hybrid Retrieval**: Combines dense and lexical search with RRF fusion and MMR diversification.
- **Observable**: Provides metrics and traces for transparency.

## Architecture

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Vector DB**: FAISS (in-process) with adapter interface
- **Retrieval**: Hybrid (dense + lexical) with RRF fusion and MMR diversification
- **LLM**: Bounded JSON-only decider (Claude 3.5 Haiku or GPT-4o-mini)
- **Safety**: Front gate (crisis detection) + back gate (evidence verification)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/coach-critique.git
   cd coach-critique
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional):
   ```bash
   export LLM_API_KEY="your-api-key"  # Required for coaching
   export DATASET_PATH="path/to/your/dataset.json"  # Default: data/combined_dataset.json
   ```

## Usage

1. Start the server:
   ```bash
   python src/main.py --host 0.0.0.0 --port 8000
   ```

2. Build the indices (first time only):
   ```bash
   curl -X POST "http://localhost:8000/index" -H "Content-Type: application/json" -d '{"dataset_path": "data/combined_dataset.json"}'
   ```

3. Search for cases:
   ```bash
   curl -X POST "http://localhost:8000/search_cases" -H "Content-Type: application/json" -d '{"query": "I feel anxious and overwhelmed", "k": 3}'
   ```

4. Generate a coaching response:
   ```bash
   curl -X POST "http://localhost:8000/coach" -H "Content-Type: application/json" -d '{"query": "I feel anxious and overwhelmed", "case_ids": [1, 2, 3]}'
   ```

5. Get metrics:
   ```bash
   curl "http://localhost:8000/metrics"
   ```

## API Endpoints

- `POST /index`: Build indices from dataset
- `POST /search_cases`: Search for cases similar to the query
- `POST /coach`: Generate a coaching response
- `GET /metrics`: Get metrics
- `GET /health`: Health check

## Dataset Format

The dataset should be in NDJSON format, with each line containing a JSON object with `Context` and `Response` fields:

```json
{"Context": "I'm feeling depressed and anxious", "Response": "It's normal to feel this way..."}
```

## Configuration

Configuration can be set via environment variables or command-line arguments:

- `DATA_DIR`: Directory for data files (default: `data`)
- `DATASET_PATH`: Path to dataset file (default: `data/combined_dataset.json`)
- `INDEX_DIR`: Directory for indices (default: `data/indices`)
- `EMBED_MODEL`: Embedding model name (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `LLM_API_KEY`: API key for LLM (required for coaching)
- `SEARCH_TIMEOUT`: Timeout for search requests in seconds (default: `10`)
- `COACH_TIMEOUT`: Timeout for coach requests in seconds (default: `20`)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This is a demonstration system only and not intended to provide medical or clinical advice. In case of crisis, please contact appropriate emergency services or crisis hotlines.

## Acknowledgements

- The design is based on best practices in retrieval-augmented generation and safety-first AI systems.
- This system prioritizes safety, evidence, and transparency over generative capabilities.
