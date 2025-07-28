import logging
import time
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ..models.schema import SearchRequest, CoachRequest, CoachResponse, SearchResult
from ..services.index_manager import IndexManager
from ..services.embedding_service import EmbeddingService
from ..services.retrieval_service import RetrievalService
from ..services.safety_service import SafetyService
from ..services.coach_service import CoachService
from ..utils.data_loader import load_dataset, save_cases

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CoachCritique API",
    description="Safety-First Coaching with Bounded LLM & Evidence Gate",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Metrics
metrics = {
    "requests": 0,
    "search_requests": 0,
    "coach_requests": 0,
    "crisis_detected": 0,
    "gate_passed": 0,
    "gate_failed": 0,
    "latencies": [],
    "search_latencies": [],
    "coach_latencies": [],
}

# Configuration
config = {
    "data_dir": os.environ.get("DATA_DIR", "data"),
    "dataset_path": os.environ.get("DATASET_PATH", "data/combined_dataset.json"),
    "cases_path": os.environ.get("CASES_PATH", "data/cases.json"),
    "index_dir": os.environ.get("INDEX_DIR", "data/indices"),
    "embed_model": os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    "llm_api_key": os.environ.get("LLM_API_KEY", None),
    "search_timeout": int(os.environ.get("SEARCH_TIMEOUT", 10)),
    "coach_timeout": int(os.environ.get("COACH_TIMEOUT", 20)),
}

# Initialize services
index_manager = IndexManager({
    "index_dir": config["index_dir"],
    "cases_path": config["cases_path"],
})



# Dependency to get services
async def get_services():
    """Get initialized services."""
    if not index_manager.is_initialized():
        try:
            # Try to load indices from disk
            logger.info("Attempting to load existing indices from disk")
            embedding_service = EmbeddingService(config["embed_model"])
            if not index_manager.load_indices(embedding_service):
                logger.error("Failed to load indices from disk")
                raise ValueError("Failed to load indices")
            logger.info("Successfully loaded indices from disk")
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise HTTPException(status_code=503, detail="Services not initialized. Please run /index endpoint first.")
    
    context_index, response_index, cases, embedding_service = index_manager.get_indices()
    
    safety_service = SafetyService()
    
    retrieval_service = RetrievalService(
        context_index=context_index,
        response_index=response_index,
        embedding_service=embedding_service,
        cases=cases
    )
    
    coach_service = CoachService(
        cases=cases,
        safety_service=safety_service,
        config={"llm_api_key": config["llm_api_key"]}
    )
    
    return {
        "retrieval_service": retrieval_service,
        "safety_service": safety_service,
        "coach_service": coach_service,
        "embedding_service": embedding_service,
        "cases": cases
    }


# Request models
class IndexRequest(BaseModel):
    dataset_path: Optional[str] = None
    max_items: Optional[int] = None


# Response models
class IndexResponse(BaseModel):
    success: bool
    message: str
    stats: Dict[str, Any]


class MetricsResponse(BaseModel):
    hit_at_3: float
    gate_pass_rate: float
    refusal_rate: float
    p50_ms: float
    p95_ms: float
    request_count: int


# Middleware for metrics
@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """Middleware to collect metrics."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Update metrics
    if request.url.path in ["/api/search_cases", "/api/coach"]:
        metrics["requests"] += 1
        metrics["latencies"].append(process_time * 1000)  # Convert to ms
        
        if request.url.path == "/api/search_cases":
            metrics["search_requests"] += 1
            metrics["search_latencies"].append(process_time * 1000)
        elif request.url.path == "/api/coach":
            metrics["coach_requests"] += 1
            metrics["coach_latencies"].append(process_time * 1000)
    
    return response


# Background task for indexing
async def build_indices_task(dataset_path: str, max_items: Optional[int] = None):
    """Background task to build indices."""
    try:
        # Load dataset
        cases = load_dataset(dataset_path, max_items)
        
        # Save cases
        save_cases(cases, config["cases_path"])
        
        # Create embedding service
        embedding_service = EmbeddingService(config["embed_model"])
        
        # Build indices
        index_manager.build_indices(cases, embedding_service)
        
        # Save indices
        index_manager.save_indices()
        
        logger.info("Indices built successfully")
    except Exception as e:
        logger.error(f"Failed to build indices: {e}")


# API endpoints
@app.post("/index", response_model=IndexResponse)
async def index(request: IndexRequest, background_tasks: BackgroundTasks):
    """Build indices from dataset."""
    dataset_path = request.dataset_path or config["dataset_path"]
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")
    
    # Start background task
    background_tasks.add_task(build_indices_task, dataset_path, request.max_items)
    
    return {
        "success": True,
        "message": f"Building indices from {dataset_path} (max_items={request.max_items}). This may take a while.",
        "stats": index_manager.get_stats() if index_manager.is_initialized() else {"initialized": False}
    }


@app.post("/api/search_cases", response_model=List[SearchResult])
async def search_cases(
    request: SearchRequest,
    services: Dict[str, Any] = Depends(get_services)
):
    """Search for cases similar to the query."""
    start_time = time.time()
    logger.info(f"Searching for cases: {request.query}")
    
    # Get services
    retrieval_service = services["retrieval_service"]
    
    # Set timeout
    try:
        # Search for cases
        results = await asyncio.wait_for(
            asyncio.to_thread(retrieval_service.search_cases, request.query, request.k, None, request.include_highlights),
            timeout=config["search_timeout"]
        )
        
        logger.info(f"Found {len(results)} cases in {time.time() - start_time:.2f}s")
        return results
    except asyncio.TimeoutError:
        logger.error(f"Search timed out after {config['search_timeout']}s")
        raise HTTPException(status_code=504, detail="Search timed out")


class CoachRequest(BaseModel):
    """Request body for coach endpoint."""
    query: str
    case_ids: List[int] = []  # Optional list of selected case IDs

@app.post("/api/coach")
async def coach(
    request: CoachRequest,
    services: Dict[str, Any] = Depends(get_services)
):
    """Generate coaching response."""
    logger.info(f"Coach request: {request.query}")
    
    try:
        # Get services
        retrieval_service = services["retrieval_service"]
        coach_service = services["coach_service"]
        
        # Get cases
        if request.case_ids and len(request.case_ids) > 0:
            logger.info(f"Using selected case IDs: {request.case_ids}")
            # Get cases from selected IDs
            cases = []
            for case_id in request.case_ids:
                case_results = retrieval_service.search_cases(
                    request.query, 
                    filter_case_ids=[case_id],
                    include_highlights=False  # Don't need highlights for coaching
                )
                if case_results:
                    cases.extend([{
                        "case_id": c.case_id,
                        "context": c.context,
                        "response": c.response,  # Include full response
                        "highlights": [h.dict() for h in c.highlights]  # Will be empty list by default
                    } for c in case_results])
        else:
            # Search for relevant cases
            search_results = retrieval_service.search_cases(
                request.query,
                include_highlights=False  # Don't need highlights for coaching
            )
            
            # Convert to dict for coach service
            cases = [{
                "case_id": c.case_id,
                "context": c.context,
                "response": c.response,  # Include full response
                "highlights": [h.dict() for h in c.highlights]  # Will be empty list by default
            } for c in search_results]
        
        # Generate coaching response with timeout
        response = await asyncio.wait_for(
            coach_service.coach(request.query, cases),
            timeout=config["coach_timeout"]
        )
        
        # Update gate metrics
        metrics["coach_requests"] += 1
        if getattr(response, "refused", False):
            metrics["gate_failed"] += 1
        else:
            metrics["gate_passed"] += 1
        
        return response.dict()
    except Exception as e:
        logger.exception(f"Error in coach endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get metrics."""
    # Calculate metrics
    hit_at_3 = 0.8  # Placeholder - would need evaluation data
    
    gate_pass_rate = 0.0
    if metrics["gate_passed"] + metrics["gate_failed"] > 0:
        gate_pass_rate = metrics["gate_passed"] / (metrics["gate_passed"] + metrics["gate_failed"])
    
    refusal_rate = 0.0
    if metrics["coach_requests"] > 0:
        refusal_rate = metrics["gate_failed"] / metrics["coach_requests"]
    
    # Calculate latency percentiles
    p50_ms = 0.0
    p95_ms = 0.0
    
    if metrics["latencies"]:
        sorted_latencies = sorted(metrics["latencies"])
        p50_idx = int(len(sorted_latencies) * 0.5)
        p95_idx = int(len(sorted_latencies) * 0.95)
        
        p50_ms = sorted_latencies[p50_idx]
        p95_ms = sorted_latencies[p95_idx]
    
    return {
        "hit_at_3": hit_at_3,
        "gate_pass_rate": gate_pass_rate,
        "refusal_rate": refusal_rate,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "request_count": metrics["requests"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check if pre-built indices exist (they should always exist in production)
    indices_exist = (
        os.path.exists(os.path.join(config["index_dir"], "index_manifest.json")) and
        os.path.exists(config["cases_path"])
    )
    
    # Show initialized=true if indices exist OR if manager is initialized
    is_ready = index_manager.is_initialized() or indices_exist
    
    return {"status": "ok", "initialized": is_ready}


@app.get("/api/cases/{case_id}")
async def get_case(
    case_id: int,
    services: Dict[str, Any] = Depends(get_services)
):
    """Get a specific case by ID."""
    cases = services["cases"]
    
    try:
        case = next(c for c in cases if c.id == case_id)
        return {
            "case_id": case.id,
            "context": case.context,
            "response": case.response,
            "response_sentences": [
                {
                    "sent_id": i,
                    "text": sentence.text,
                    "start": sentence.start,
                    "end": sentence.end
                }
                for i, sentence in enumerate(case.response_sentences)
            ]
        }
    except StopIteration:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    except Exception as e:
        logger.error(f"Error getting case {case_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error")


if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 