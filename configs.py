# enhanced_main_optimized.py - Multi-threaded PDF Q&A with Neo4j Integration and Rate Limiting Fixes

import os
import asyncio
import aiohttp
import fitz  # PyMuPDF
import spacy
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from graphDatabase.neo4j import GraphDatabase
from groq import Groq
import json
import uuid
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass, field
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
import uvicorn
from datetime import datetime
import logging
import traceback
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
import time
import numpy as np
import multiprocessing
import hashlib
from asyncio import Semaphore
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class Config:
    # System configuration - REDUCED for rate limiting
    TOTAL_THREADS: int = min(3, max(1, multiprocessing.cpu_count() - 1))  # Max 3 threads
    CHUNK_THREADS: int = max(1, (multiprocessing.cpu_count() - 1) // 2)  # Half for chunking
    QUERY_THREADS: int = min(3, max(1, multiprocessing.cpu_count() - 1))  # Max 3 concurrent queries
    
    # Multiple Groq API Keys (one per thread) - FIXED
    GROQ_API_KEYS: List[str] = field(default_factory=lambda: [
        os.getenv(f"GROQ_API_KEY_{i+1}", os.getenv("GROQ_API_KEY"))
        for i in range(min(3, max(1, multiprocessing.cpu_count() - 1)))  # Max 3 API keys
    ])
    
    # Database settings
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")
    
    # Model settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "7"))
    REDIS_AVAILABLE = False  # Enable/disable Redis caching
    REDIS_HOST: str = os.getenv("REDIS_HOST", " ")
    REDIS_URL: str = os.getenv("REDIS_URL", " ")
    REDIS_PORT : int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD : str = os.getenv("REDIS_PASSWORD", " ")
    REDIS_DB : str= os.getenv("REDIS_DB", "0")
    REDIS_SSL = os.getenv('REDIS_SSL', 'false').lower() == 'true'
    def __post_init__(self):
        # Filter out None values from API keys
        self.GROQ_API_KEYS = [key for key in self.GROQ_API_KEYS if key]
        if not self.GROQ_API_KEYS:
            raise ValueError("At least one GROQ_API_KEY is required")
        logger.info(f"Configured {len(self.GROQ_API_KEYS)} Groq API keys for {self.TOTAL_THREADS} threads")