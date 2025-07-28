# Embedding for CoachCritique

This document provides instructions for creating embeddings for CoachCritique.

## Overview

The embedding process creates the necessary vector indices for CoachCritique's retrieval functionality:

1. Load the counseling Q&A dataset
2. Create embeddings for contexts and response sentences
3. Build FAISS indices for both
4. Save the indices and manifests

## Prerequisites

- Python 3.9+
- pip

## Installation

1. Install the required packages:

```bash
python -m venv venv_py39
source venv_py39/bin/activate  # On Windows: venv_py39\Scripts\activate
pip install -r requirements.txt
```

## Creating Indices

Run the index creation script:

```bash
python create_indices_if_needed.py
```

This will:
- Load the dataset from `data/cases.json` (or fallback to `data/combined_dataset.json`)
- Build indices using the `sentence-transformers/all-MiniLM-L6-v2` model
- Save the indices to `data/indices/`

## Running the Full Application

After creating indices, you can run the full application:

```bash
./run_with_indices.sh
```

This script will:
1. Check if indices exist and create them if needed
2. Start the FastAPI backend server
3. Allow you to connect with the frontend application

## Key Parameters

The system uses the following key parameters as specified in the architecture document:
- **Index A**: Context index (FlatIP)
- **Index B**: Response sentence index (FlatIP)
- Top-k = 3 (return 3 most similar cases)
- Sentence highlights = 3 (return 3 most relevant sentences per case)

## Integration

The indices are automatically used by:
1. The retrieval service for semantic search
2. The coaching service for evidence-based responses 