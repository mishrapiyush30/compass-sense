#!/bin/bash
# Run the backend server with index checking

# Activate Python 3.9 virtual environment
source venv_py39/bin/activate

# Make sure PYTHONPATH includes the src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Load environment variables from .env file
# Create a .env file with your LLM_API_KEY before running this script
# Example: LLM_API_KEY=your_api_key_here
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "Warning: .env file not found. Make sure LLM_API_KEY is set in your environment."
fi

# Check if indices exist and create them if needed
echo "Checking if indices exist..."
python3 create_indices_if_needed.py

# Run the FastAPI server with uvicorn
echo "Starting backend server..."
python -m uvicorn src.backend.api.app:app --host 0.0.0.0 --port 8080 --reload 