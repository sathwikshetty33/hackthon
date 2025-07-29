# main.py - Ultra-Enhanced PDF Q&A FastAPI System
import os
import asyncio
import aiohttp
import fitz  # PyMuPDF
import spacy
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from neo4j import GraphDatabase
from groq import Groq
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
import uvicorn
from datetime import datetime
import logging
import traceback
from contextlib import asynccontextmanager
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import hashlib
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Pydantic models for API
class QuestionRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

class DetailedAnswerResponse(BaseModel):
    answers: List[Dict[str, Any]]

@dataclass
class Config:
    # API Keys from environment
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")
    
    # Optimized model settings for speed and accuracy
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))  # Optimized for balance
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))  # Focused context
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    
    # Performance settings
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))
    CACHE_SIZE: int = int(os.getenv("CACHE_SIZE", "128"))

class UltraEnhancedPDFQASystem:
    def __init__(self):
        self.config = Config()
        self._validate_config()
        self._initialize_components()
        self._setup_caching()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        self._document_cache = {}
        self._embedding_cache = {}
        self.lock = threading.Lock()

    def _validate_config(self):
        """Validate that all required environment variables are set"""
        required_vars = [("GROQ_API_KEY", self.config.GROQ_API_KEY)]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        logger.info("All required environment variables loaded successfully")

    def _setup_caching(self):
        """Setup LRU caches for performance"""
        # Cache for embeddings
        self.get_embedding = lru_cache(maxsize=self.config.CACHE_SIZE)(self._get_embedding_uncached)
        # Cache for document parsing
        self.parse_pdf_cached = lru_cache(maxsize=32)(self._parse_pdf_uncached)

    def _initialize_components(self):
        """Initialize all components with performance optimizations"""
        logger.info("Initializing Ultra-Enhanced PDF Q&A System...")
        
        # Load spaCy model with optimizations
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Disable unused components
            logger.info("spaCy model loaded with optimizations")
        except OSError:
            logger.warning("spaCy model not found, using fallback text processing")
            self.nlp = None
        
        # Initialize embedding model with optimizations
        logger.info("Loading optimized sentence transformer...")
        try:
            # Use faster model for better speed/accuracy balance
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_model.eval()  # Set to eval mode
            self.embedding_dim = 384
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        logger.info("Embedding model loaded with optimizations")
        
        # Initialize Groq client with retry logic
        logger.info("Connecting to Groq API...")
        self.groq_client = Groq(api_key=self.config.GROQ_API_KEY)
        logger.info("Groq client initialized")
        
        # Initialize optional components
        self._initialize_optional_components()
        
        logger.info("All components initialized successfully!")

    def _initialize_optional_components(self):
        """Initialize optional Qdrant and Neo4j components"""
        # Initialize Qdrant client (optional)
        self.qdrant_client = None
        if self.config.QDRANT_URL and self.config.QDRANT_API_KEY:
            try:
                logger.info("Connecting to Qdrant...")
                self.qdrant_client = QdrantClient(
                    url=self.config.QDRANT_URL,
                    api_key=self.config.QDRANT_API_KEY,
                )
                logger.info("Qdrant client initialized")
            except Exception as e:
                logger.warning(f"Qdrant connection failed: {e}. Using in-memory search.")
        
        # Initialize Neo4j driver (optional)
        self.neo4j_driver = None
        if self.config.NEO4J_URI and self.config.NEO4J_PASSWORD:
            try:
                logger.info("Connecting to Neo4j...")
                self.neo4j_driver = GraphDatabase.driver(
                    self.config.NEO4J_URI,
                    auth=(self.config.NEO4J_USERNAME, self.config.NEO4J_PASSWORD)
                )
                logger.info("Neo4j driver initialized")
            except Exception as e:
                logger.warning(f"Neo4j connection failed: {e}. Using basic search.")

    def _get_document_hash(self, url: str) -> str:
        """Generate hash for document caching"""
        return hashlib.md5(url.encode()).hexdigest()

    async def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL with caching and optimization"""
        doc_hash = self._get_document_hash(url)
        
        # Check cache first
        if doc_hash in self._document_cache:
            logger.info(f"Using cached PDF for: {url}")
            return self._document_cache[doc_hash]
        
        logger.info(f"Downloading PDF from: {url}")
        try:
            timeout = aiohttp.ClientTimeout(total=120, connect=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(str(url)) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Failed to download PDF: HTTP {response.status}"
                        )
                    
                    content = await response.read()
                    
                    # Validate PDF content
                    if not content.startswith(b'%PDF'):
                        raise HTTPException(
                            status_code=400, 
                            detail="Downloaded file is not a valid PDF"
                        )
                    
                    # Cache the document
                    self._document_cache[doc_hash] = content
                    logger.info(f"PDF downloaded and cached successfully ({len(content)} bytes)")
                    return content
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error downloading PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to download PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

    def _parse_pdf_uncached(self, pdf_hash: str, pdf_bytes: bytes) -> List[Dict]:
        """Parse PDF implementation for caching"""
        return self._parse_pdf_content(pdf_bytes)

    def parse_pdf(self, pdf_bytes: bytes) -> List[Dict]:
        """Parse PDF with caching"""
        pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
        return self.parse_pdf_cached(pdf_hash, pdf_bytes)

    def _parse_pdf_content(self, pdf_bytes: bytes) -> List[Dict]:
        """Parse PDF and extract text with enhanced metadata and speed optimizations"""
        logger.info("Parsing PDF...")
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text more efficiently
                text = page.get_text("text")  # Use plain text extraction for speed
                
                # Skip empty pages
                if not text.strip() or len(text.strip()) < 50:
                    continue
                
                # Fast structure analysis
                sections = self._fast_extract_sections(text)
                tables = self._fast_extract_tables(page)
                
                pages.append({
                    'page_number': page_num + 1,
                    'text': text,
                    'sections': sections,
                    'tables': tables,
                    'word_count': len(text.split()),
                    'char_count': len(text)
                })
            
            doc.close()
            logger.info(f"PDF parsed successfully ({len(pages)} pages)")
            return pages
            
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")

    def _fast_extract_sections(self, text: str) -> List[str]:
        """Fast section extraction using regex patterns"""
        sections = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Quick heuristics for section headers
            if (len(line) > 5 and len(line) < 150 and
                (line.isupper() or 
                 re.match(r'^\d+\.?\s+[A-Z]', line) or
                 re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5}:?\s*$', line))):
                sections.append(line)
        
        return sections[:10] if sections else ["Document Content"]  # Limit sections

    def _fast_extract_tables(self, page) -> List[Dict]:
        """Fast table extraction with limits"""
        tables = []
        try:
            table_data = page.find_tables()
            for i, table in enumerate(table_data[:3]):  # Limit to 3 tables per page
                extracted = table.extract()
                if extracted and len(extracted) > 1:
                    tables.append({
                        'table_id': i,
                        'rows': len(extracted),
                        'cols': len(extracted[0]) if extracted else 0
                    })
        except:
            pass
        return tables

    def optimized_chunk_document(self, pages: List[Dict]) -> List[Dict]:
        """Create optimized semantic chunks with better performance"""
        logger.info("Creating optimized document chunks...")
        chunks = []
        
        for page in pages:
            text = page['text']
            page_num = page['page_number']
            sections = page['sections']
            
            # Use more efficient chunking
            current_section = sections[0] if sections else "Document Content"
            
            # Split by double newlines and periods for better semantic boundaries
            sentences = re.split(r'[.!?]+\s*(?=\n|\s[A-Z])', text)
            
            current_chunk = ""
            chunk_id = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip very short sentences
                    continue
                
                # Check if adding sentence exceeds chunk size
                if len(current_chunk) + len(sentence) > self.config.CHUNK_SIZE and current_chunk:
                    # Create chunk
                    chunks.append({
                        'id': f"{page_num}_{chunk_id}",
                        'text': current_chunk.strip(),
                        'page_number': page_num,
                        'section': current_section,
                        'word_count': len(current_chunk.split()),
                        'metadata': {
                            'source': 'pdf',
                            'page': page_num,
                            'section': current_section,
                            'chunk_index': chunk_id
                        }
                    })
                    
                    # Start new chunk with overlap
                    overlap_size = min(self.config.CHUNK_OVERLAP, len(current_chunk) // 3)
                    current_chunk = current_chunk[-overlap_size:] + " " + sentence
                    chunk_id += 1
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            # Add remaining chunk
            if current_chunk.strip() and len(current_chunk.strip()) > 50:
                chunks.append({
                    'id': f"{page_num}_{chunk_id}",
                    'text': current_chunk.strip(),
                    'page_number': page_num,
                    'section': current_section,
                    'word_count': len(current_chunk.split()),
                    'metadata': {
                        'source': 'pdf',
                        'page': page_num,
                        'section': current_section,
                        'chunk_index': chunk_id
                    }
                })
        
        logger.info(f"Created {len(chunks)} optimized chunks")
        return chunks

    def _get_embedding_uncached(self, text: str) -> List[float]:
        """Get embedding for text (uncached version)"""
        return self.embedding_model.encode([text], normalize_embeddings=True)[0].tolist()

    def batch_create_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Create embeddings in optimized batches"""
        logger.info("Creating embeddings in batches...")
        texts = [chunk['text'] for chunk in chunks]
        
        try:
            # Process in batches for memory efficiency
            all_embeddings = []
            batch_size = self.config.BATCH_SIZE
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                all_embeddings.extend(batch_embeddings)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = all_embeddings[i].tolist()
            
            logger.info(f"Created embeddings for {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")

    def ultra_fast_semantic_search(self, question: str, chunks: List[Dict], top_k: int = None) -> List[Dict]:
        """Ultra-fast semantic search with numpy optimizations"""
        if top_k is None:
            top_k = self.config.MAX_CONTEXT_CHUNKS
        
        try:
            # Get question embedding
            question_embedding = self.get_embedding(question)
            question_vector = np.array(question_embedding).reshape(1, -1)
            
            # Convert chunk embeddings to numpy array
            chunk_embeddings = np.array([chunk['embedding'] for chunk in chunks])
            
            # Compute cosine similarities in batch
            similarities = cosine_similarity(question_vector, chunk_embeddings).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Filter by similarity threshold
            filtered_indices = [idx for idx in top_indices 
                             if similarities[idx] >= self.config.SIMILARITY_THRESHOLD]
            
            # If no chunks meet threshold, take top 2
            if not filtered_indices:
                filtered_indices = top_indices[:2].tolist()
            
            # Return top chunks with similarity scores
            top_chunks = []
            for idx in filtered_indices:
                chunk = chunks[idx].copy()
                chunk['similarity_score'] = float(similarities[idx])
                top_chunks.append(chunk)
            
            return top_chunks
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return chunks[:top_k]

    def generate_optimized_answer(self, question: str, context_chunks: List[Dict]) -> Dict:
        """Generate high-quality answers with optimized prompting"""
        
        # Prepare focused context
        context_parts = []
        total_context_length = 0
        max_context_length = 2000  # Limit context for speed
        
        for i, chunk in enumerate(context_chunks, 1):
            chunk_text = chunk['text']
            if total_context_length + len(chunk_text) > max_context_length:
                # Truncate if too long
                remaining_length = max_context_length - total_context_length
                chunk_text = chunk_text[:remaining_length] + "..."
                context_parts.append(f"[Source {i} - Page {chunk['page_number']}]: {chunk_text}")
                break
            
            context_parts.append(f"[Source {i} - Page {chunk['page_number']}]: {chunk_text}")
            total_context_length += len(chunk_text)
        
        context_text = "\n\n".join(context_parts)
        
        # Optimized prompt for better accuracy and speed
        prompt = f"""Based on the following document excerpts, provide a precise and complete answer to the question. Focus on accuracy and include specific details like numbers, timeframes, and conditions when available.

DOCUMENT EXCERPTS:
{context_text}

QUESTION: {question}

ANSWER REQUIREMENTS:
- Use only information from the provided excerpts
- Be specific with numbers, periods, percentages, and conditions
- If the exact information isn't available, state "The specific information is not provided in the document"
- Provide a complete answer in 1-3 sentences
- Be direct and factual

ANSWER:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",  # Fast and accurate model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,  # Limit for speed
                temperature=0.1,  # Low temperature for consistency
                top_p=0.9,
                stop=None
            )
            
            answer_text = response.choices[0].message.content.strip()
            
            # Calculate confidence based on similarity scores and answer quality
            avg_similarity = sum(chunk.get('similarity_score', 0) for chunk in context_chunks) / len(context_chunks)
            
            # Boost confidence if answer contains specific details
            detail_indicators = ['%', 'days', 'months', 'years', 'hours', '$', 'rupees', 'limit', 'maximum', 'minimum']
            has_details = any(indicator in answer_text.lower() for indicator in detail_indicators)
            confidence_boost = 0.1 if has_details else 0
            
            confidence = min(avg_similarity + confidence_boost, 1.0)
            
            return {
                'question': question,
                'answer': answer_text,
                'confidence': confidence,
                'source_count': len(context_chunks),
                'sources': [
                    {
                        'page': chunk['page_number'],
                        'section': chunk.get('section', 'Unknown'),
                        'similarity_score': chunk.get('similarity_score', 0)
                    }
                    for chunk in context_chunks[:2]  # Limit sources for clean output
                ]
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'question': question,
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 0.0,
                'source_count': 0,
                'sources': []
            }

    async def process_questions_ultra_fast(self, pdf_url: str, questions: List[str]) -> List[str]:
        """Ultra-fast processing pipeline with parallel processing"""
        logger.info(f"Processing {len(questions)} questions for PDF: {pdf_url}")
        
        try:
            # Step 1: Download and parse PDF (cached)
            pdf_bytes = await self.download_pdf(pdf_url)
            pages = self.parse_pdf(pdf_bytes)
            
            if not pages:
                raise HTTPException(status_code=400, detail="No readable content found in PDF")
            
            # Step 2: Create optimized chunks
            chunks = self.optimized_chunk_document(pages)
            
            if not chunks:
                raise HTTPException(status_code=400, detail="Could not create document chunks")
            
            # Step 3: Create embeddings in batches
            chunks = self.batch_create_embeddings(chunks)
            
            # Step 4: Process questions in parallel (limited concurrency)
            semaphore = asyncio.Semaphore(3)  # Limit concurrent API calls
            
            async def process_single_question(question: str, index: int) -> Tuple[int, str]:
                async with semaphore:
                    try:
                        # Run search in thread pool to avoid blocking
                        loop = asyncio.get_event_loop()
                        relevant_chunks = await loop.run_in_executor(
                            self.thread_pool,
                            self.ultra_fast_semantic_search,
                            question,
                            chunks
                        )
                        
                        # Generate answer
                        result = self.generate_optimized_answer(question, relevant_chunks)
                        
                        logger.info(f"Question {index + 1}/{len(questions)} completed "
                                  f"(confidence: {result['confidence']:.3f})")
                        
                        return index, result['answer']
                        
                    except Exception as e:
                        logger.error(f"Error processing question {index + 1}: {e}")
                        return index, f"Error processing question: {str(e)}"
            
            # Process all questions concurrently
            tasks = [
                process_single_question(question, i) 
                for i, question in enumerate(questions)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Sort results by original question order
            sorted_results = sorted(
                [(i, answer) for i, answer in results if not isinstance(answer, Exception)],
                key=lambda x: x[0]
            )
            
            answers = [answer for _, answer in sorted_results]
            
            # Handle any exceptions
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
            
            logger.info(f"Completed processing {len(answers)} questions successfully")
            return answers
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in ultra-fast processing pipeline: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    async def process_questions_detailed(self, pdf_url: str, questions: List[str]) -> List[Dict]:
        """Process questions with detailed responses"""
        logger.info(f"Processing {len(questions)} questions with detailed responses")
        
        try:
            # Download and parse PDF
            pdf_bytes = await self.download_pdf(pdf_url)
            pages = self.parse_pdf(pdf_bytes)
            
            # Create chunks and embeddings
            chunks = self.optimized_chunk_document(pages)
            chunks = self.batch_create_embeddings(chunks)
            
            # Process questions with detailed results
            detailed_answers = []
            for i, question in enumerate(questions):
                logger.info(f"Processing detailed question {i + 1}/{len(questions)}")
                
                relevant_chunks = self.ultra_fast_semantic_search(question, chunks)
                result = self.generate_optimized_answer(question, relevant_chunks)
                detailed_answers.append(result)
            
            return detailed_answers
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in detailed processing: {e}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    def clear_cache(self):
        """Clear all caches"""
        with self.lock:
            self._document_cache.clear()
            self._embedding_cache.clear()
            # Clear LRU caches
            self.get_embedding.cache_clear()
            self.parse_pdf_cached.cache_clear()
        logger.info("All caches cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'document_cache_size': len(self._document_cache),
            'embedding_cache_info': self.get_embedding.cache_info()._asdict(),
            'pdf_cache_info': self.parse_pdf_cached.cache_info()._asdict()
        }

    def close_connections(self):
        """Close database connections and cleanup"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Neo4j connection closed")
        
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            logger.info("Thread pool shutdown")
        
        self.clear_cache()

# Global QA system instance
qa_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global qa_system
    logger.info("Starting up Ultra-Enhanced PDF Q&A System...")
    qa_system = UltraEnhancedPDFQASystem()
    yield
    # Shutdown
    logger.info("Shutting down Ultra-Enhanced PDF Q&A System...")
    if qa_system:
        qa_system.close_connections()

# Initialize FastAPI app
app = FastAPI(
    title="Ultra-Enhanced PDF Q&A API",
    description="Ultra-fast and accurate PDF Question-Answering System",
    version="3.0.0",
    lifespan=lifespan
)

@app.post("/ask", response_model=AnswerResponse)
async def ask_questions(request: QuestionRequest):
    """
    Process questions against a PDF document with ultra-fast processing
    
    - **documents**: URL to the PDF document
    - **questions**: List of questions to ask about the document
    """
    try:
        logger.info(f"Received request with {len(request.questions)} questions")
        start_time = datetime.now()
        
        # Process questions with ultra-fast pipeline
        answers = await qa_system.process_questions_ultra_fast(
            str(request.documents), 
            request.questions
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Request completed in {processing_time:.2f} seconds")
        
        return AnswerResponse(answers=answers)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/ask-detailed", response_model=DetailedAnswerResponse)
async def ask_questions_detailed(request: QuestionRequest):
    """
    Process questions against a PDF document with detailed response including confidence and sources
    """
    try:
        logger.info(f"Received detailed request with {len(request.questions)} questions")
        start_time = datetime.now()
        
        detailed_answers = await qa_system.process_questions_detailed(
            str(request.documents), 
            request.questions
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Detailed request completed in {processing_time:.2f} seconds")
        
        return DetailedAnswerResponse(answers=detailed_answers)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint with system statistics"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_stats": qa_system.get_cache_stats() if qa_system else {},
        "system_info": {
            "max_workers": qa_system.config.MAX_WORKERS if qa_system else 0,
            "chunk_size": qa_system.config.CHUNK_SIZE if qa_system else 0,
            "embedding_model": qa_system.config.EMBEDDING_MODEL if qa_system else "unknown"
        }
    }

@app.post("/clear-cache")
async def clear_cache():
    """Clear system caches"""
    if qa_system:
        qa_system.clear_cache()
        return {"message": "Caches cleared successfully"}
    return {"message": "System not initialized"}

@app.get("/")
async def root():