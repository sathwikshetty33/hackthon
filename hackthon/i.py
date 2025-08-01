from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import uvicorn
from datetime import datetime
import multiprocessing
from hackthon.configs import *
from hackthon.models import *
from hackthon.vectorDatabase.quadrantdb import *
from hackthon.graphDatabase.neo4j import *
from hackthon.SessionManager.ThreadSession import *
from hackthon.llogging import *
from contextlib import asynccontextmanager
logger = setup_logger()
# Load environment variables
load_dotenv()

# Pydantic models for API



# Global system instance
qa_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_system
    logger.info("Starting Rate-Limited Multi-Thread PDF Q&A System...")
    qa_system = MultiThreadSessionSystem()
    yield
    logger.info("Shutting down system...")
    if qa_system:
        qa_system.close_connections()

# FastAPI app
app = FastAPI(
    title="Rate-Limited Multi-Thread PDF Q&A API",
    description=f"Rate-limited multi-threaded PDF Q&A system with {min(3, max(1, multiprocessing.cpu_count() - 1))} threads",
    version="6.1.0",  # Updated version number
    lifespan=lifespan
)

@app.post("/hackrx/run")
async def hackrx_run(request: QuestionRequest):
    """HackRX testing endpoint with rate limiting"""
    try:
        logger.info(f"HackRX endpoint: Processing {len(request.questions)} questions with rate limiting")
        
        answers = await qa_system.process_questions_rate_limited(
            str(request.documents),
            request.questions
        )
        
        return {"answers": answers}
        
    except Exception as e:
        logger.error(f"HackRX endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check with system configuration"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_config": {
            "total_threads": qa_system.config.TOTAL_THREADS if qa_system else "Unknown",
            "chunk_threads": qa_system.config.CHUNK_THREADS if qa_system else "Unknown",
            "query_threads": qa_system.config.QUERY_THREADS if qa_system else "Unknown",
            "groq_api_keys": len(qa_system.config.GROQ_API_KEYS) if qa_system else 0,
            "vector_db": "Qdrant",
            "rate_limiting": "Enabled",
            "cpu_count": multiprocessing.cpu_count()
        }
    }

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Rate-Limited Multi-Thread PDF Q&A API",
        "version": "6.1.0",
        "features": [
            f"{min(3, max(1, multiprocessing.cpu_count() - 1))}-thread rate-limited processing",
            "Multiple Groq API keys with throttling",
            "Enhanced Qdrant vector database storage",
            "Exponential backoff retry logic",
            "Request semaphore limiting",
            "Token usage optimization",
            "Fixed question_index error"
        ],
        "optimizations": {
            "max_concurrent_requests": 1,  # Updated
            "min_request_interval": "4.0s",  # Updated
            "max_tokens_per_request": 600,
            "retry_attempts": 3
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "i:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
