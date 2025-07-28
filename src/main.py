#!/usr/bin/env python
"""Main entry point for CoachCritique."""

import os
import sys
import argparse
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the CoachCritique application."""
    # Change the working directory to the project root
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    project_root = os.path.dirname(script_dir)
    # Change the working directory
    os.chdir(project_root)
    logger.info(f"Changed working directory to: {os.getcwd()}")
    
    # Add the project root to the Python path so modules can be found
    sys.path.insert(0, project_root)
    
    parser = argparse.ArgumentParser(description="CoachCritique - Safety-First Coaching")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--dataset", default=None, help="Path to dataset file")
    parser.add_argument("--llm-api-key", default=None, help="LLM API key")
    
    args = parser.parse_args()
    
    # Set environment variables
    if args.data_dir:
        os.environ["DATA_DIR"] = args.data_dir
    
    if args.dataset:
        os.environ["DATASET_PATH"] = args.dataset
    
    if args.llm_api_key:
        os.environ["LLM_API_KEY"] = args.llm_api_key
    
    # Run the FastAPI app
    logger.info(f"Starting CoachCritique on {args.host}:{args.port}")
    uvicorn.run(
        "src.backend.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main() 