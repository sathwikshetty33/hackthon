# main.py - Enhanced PDF Q&A FastAPI System
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
from dataclasses import dataclass
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
import uvicorn
from datetime import datetime
import logging
import traceback
from contextlib import asynccontextmanager

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
    
    # Model settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))  # Increased for better context
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))  # Increased overlap
    MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "7"))  # More context

class EnhancedPDFQASystem:
    def __init__(self):
        self.config = Config()
        self._validate_config()
        self._initialize_components()

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

    def _initialize_components(self):
        """Initialize all components"""
        logger.info("Initializing PDF Q&A System...")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded")
        except OSError:
            logger.warning("spaCy model not found, using fallback text processing")
            self.nlp = None
        
        # Initialize embedding model with better model
        logger.info("Loading sentence transformer...")
        try:
            # Use a more accurate model
            self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            self.embedding_dim = 768
        except:
            # Fallback to smaller model
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            self.embedding_dim = 384
        logger.info("Embedding model loaded")
        
        # Initialize Groq client
        logger.info("Connecting to Groq API...")
        self.groq_client = Groq(api_key=self.config.GROQ_API_KEY)
        logger.info("Groq client initialized")
        
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
        
        logger.info("All components initialized successfully!")

    async def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL asynchronously"""
        logger.info(f"Downloading PDF from: {url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(str(url), timeout=aiohttp.ClientTimeout(total=60)) as response:
                    response.raise_for_status()
                    content = await response.read()
                    logger.info(f"PDF downloaded successfully ({len(content)} bytes)")
                    return content
        except Exception as e:
            logger.error(f"Failed to download PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

    def parse_pdf(self, pdf_bytes: bytes) -> List[Dict]:
        """Parse PDF and extract text with enhanced metadata"""
        logger.info("Parsing PDF...")
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Skip empty pages
                if not text.strip():
                    continue
                
                # Extract structure information
                blocks = page.get_text("dict")
                sections = self._extract_sections(blocks)
                
                # Extract tables and lists
                tables = self._extract_tables(page)
                lists = self._extract_lists(text)
                
                pages.append({
                    'page_number': page_num + 1,
                    'text': text,
                    'sections': sections,
                    'tables': tables,
                    'lists': lists,
                    'word_count': len(text.split())
                })
            
            doc.close()
            logger.info(f"PDF parsed successfully ({len(pages)} pages)")
            return pages
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")

    def _extract_sections(self, blocks_dict) -> List[str]:
        """Extract section headings with improved heuristics"""
        sections = []
        for block in blocks_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        size = span.get("size", 0)
                        flags = span.get("flags", 0)
                        
                        # Enhanced heuristic for headings
                        is_bold = flags & 2**4
                        is_large = size > 12
                        is_title_case = text.istitle() or text.isupper()
                        is_reasonable_length = 5 < len(text) < 200
                        
                        if (is_reasonable_length and 
                            (is_bold or is_large or is_title_case) and
                            not text.endswith('.') and
                            len(text.split()) < 15):
                            sections.append(text)
        
        return sections if sections else ["Document Content"]

    def _extract_tables(self, page) -> List[Dict]:
        """Extract table information"""
        tables = []
        try:
            table_data = page.find_tables()
            for i, table in enumerate(table_data):
                if table.extract():
                    tables.append({
                        'table_id': i,
                        'data': table.extract()[:5]  # First 5 rows to avoid too much data
                    })
        except:
            pass
        return tables

    def _extract_lists(self, text: str) -> List[str]:
        """Extract bullet points and numbered lists"""
        lists = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Check for bullet points or numbered lists
            if (re.match(r'^[•·▪▫◦‣⁃]\s+', line) or 
                re.match(r'^\d+\.\s+', line) or
                re.match(r'^[a-zA-Z]\.\s+', line)):
                lists.append(line)
        return lists

    def smart_chunk_document(self, pages: List[Dict]) -> List[Dict]:
        """Create intelligent semantic chunks"""
        logger.info("Creating intelligent document chunks...")
        chunks = []
        
        for page in pages:
            text = page['text']
            page_num = page['page_number']
            sections = page['sections']
            
            # Split by paragraphs first
            paragraphs = re.split(r'\n\s*\n', text)
            
            current_chunk = ""
            current_section = sections[0] if sections else "Document Content"
            chunk_sentences = []
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Split paragraph into sentences
                sentences = re.split(r'[.!?]+', paragraph)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence or len(sentence) < 10:
                        continue
                    
                    # Check if adding sentence exceeds chunk size
                    potential_chunk = current_chunk + " " + sentence
                    
                    if len(potential_chunk) > self.config.CHUNK_SIZE and current_chunk:
                        # Create chunk with overlap
                        chunks.append({
                            'id': str(uuid.uuid4()),
                            'text': current_chunk.strip(),
                            'page_number': page_num,
                            'section': current_section,
                            'sentence_count': len(chunk_sentences),
                            'metadata': {
                                'source': 'pdf',
                                'page': page_num,
                                'section': current_section,
                                'word_count': len(current_chunk.split())
                            }
                        })
                        
                        # Start new chunk with overlap
                        overlap_sentences = chunk_sentences[-2:] if len(chunk_sentences) >= 2 else chunk_sentences
                        current_chunk = " ".join(overlap_sentences) + " " + sentence
                        chunk_sentences = overlap_sentences + [sentence]
                    else:
                        current_chunk = potential_chunk
                        chunk_sentences.append(sentence)
            
            # Add remaining chunk
            if current_chunk.strip():
                chunks.append({
                    'id': str(uuid.uuid4()),
                    'text': current_chunk.strip(),
                    'page_number': page_num,
                    'section': current_section,
                    'sentence_count': len(chunk_sentences),
                    'metadata': {
                        'source': 'pdf',
                        'page': page_num,
                        'section': current_section,
                        'word_count': len(current_chunk.split())
                    }
                })
        
        logger.info(f"Created {len(chunks)} intelligent chunks")
        return chunks

    def create_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Create embeddings with progress tracking"""
        logger.info("Creating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        
        try:
            embeddings = self.embedding_model.encode(
                texts, 
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True
            )
            
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = embeddings[i].tolist()
            
            logger.info("Embeddings created successfully")
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")
        
        return chunks

    def semantic_search(self, question: str, chunks: List[Dict], top_k: int = None) -> List[Dict]:
        """Perform semantic search on chunks"""
        if top_k is None:
            top_k = self.config.MAX_CONTEXT_CHUNKS
        
        try:
            # Create question embedding
            question_embedding = self.embedding_model.encode([question], normalize_embeddings=True)[0]
            
            # Calculate similarities
            similarities = []
            for i, chunk in enumerate(chunks):
                chunk_embedding = chunk['embedding']
                # Cosine similarity (since embeddings are normalized)
                similarity = sum(a * b for a, b in zip(question_embedding, chunk_embedding))
                similarities.append((similarity, i))
            
            # Sort by similarity
            similarities.sort(reverse=True)
            
            # Return top-k chunks with similarity scores
            top_chunks = []
            for similarity, idx in similarities[:top_k]:
                chunk = chunks[idx].copy()
                chunk['similarity_score'] = float(similarity)
                top_chunks.append(chunk)
            
            return top_chunks
        
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return chunks[:top_k]  # Fallback to first k chunks

    def enhanced_answer_generation(self, question: str, context_chunks: List[Dict]) -> Dict:
        """Generate high-quality answers with improved prompting"""
        
        # Prepare context with better formatting
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Context {i} - Page {chunk['page_number']}, Section: {chunk['section']}]\n{chunk['text']}"
            )
        
        context_text = "\n\n".join(context_parts)
        
        # Enhanced prompt with better instructions
        prompt = f"""You are an expert document analyst. Your task is to provide accurate, comprehensive answers based solely on the provided context from a policy document.

CONTEXT:
{context_text}

QUESTION: {question}

INSTRUCTIONS:
1. Read the question carefully and identify what specific information is being asked
2. Search through ALL the provided context for relevant information
3. Provide a complete, accurate answer based ONLY on the information in the context
4. If the exact answer is not available, state "The specific information is not available in the provided document sections"
5. Be precise with numbers, dates, percentages, and specific terms
6. Include relevant details that directly answer the question
7. Do not make assumptions or add information not present in the context
8. Structure your answer clearly and concisely

ANSWER:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",  # Better model for accuracy
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.1,  # Low temperature for consistency
                top_p=0.9
            )
            
            answer_text = response.choices[0].message.content.strip()
            
            # Calculate confidence based on similarity scores
            avg_similarity = sum(chunk.get('similarity_score', 0) for chunk in context_chunks) / len(context_chunks)
            confidence = min(avg_similarity * 1.2, 1.0)  # Boost confidence slightly
            
            return {
                'question': question,
                'answer': answer_text,
                'confidence': confidence,
                'sources': [
                    {
                        'page': chunk['page_number'],
                        'section': chunk['section'],
                        'similarity_score': chunk.get('similarity_score', 0)
                    }
                    for chunk in context_chunks[:3]
                ]
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'question': question,
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 0.0,
                'sources': []
            }

    async def process_questions(self, pdf_url: str, questions: List[str]) -> List[str]:
        """Main processing pipeline optimized for speed and accuracy"""
        logger.info(f"Processing {len(questions)} questions for PDF: {pdf_url}")
        
        try:
            # Step 1: Download and parse PDF
            pdf_bytes = await self.download_pdf(pdf_url)
            pages = self.parse_pdf(pdf_bytes)
            
            # Step 2: Create intelligent chunks
            chunks = self.smart_chunk_document(pages)
            
            # Step 3: Create embeddings
            chunks = self.create_embeddings(chunks)
            
            # Step 4: Process all questions
            answers = []
            for i, question in enumerate(questions, 1):
                logger.info(f"Processing question {i}/{len(questions)}: {question[:50]}...")
                
                # Semantic search for relevant chunks
                relevant_chunks = self.semantic_search(question, chunks)
                
                # Generate answer
                result = self.enhanced_answer_generation(question, relevant_chunks)
                answers.append(result['answer'])
                
                logger.info(f"Question {i} completed with confidence: {result['confidence']:.3f}")
            
            return answers
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    def close_connections(self):
        """Close database connections"""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Neo4j connection closed")

# Global QA system instance
qa_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global qa_system
    logger.info("Starting up PDF Q&A System...")
    qa_system = EnhancedPDFQASystem()
    yield
    # Shutdown
    logger.info("Shutting down PDF Q&A System...")
    if qa_system:
        qa_system.close_connections()

# Initialize FastAPI app
app = FastAPI(
    title="PDF Q&A API",
    description="Enhanced PDF Question-Answering System with high accuracy",
    version="2.0.0",
    lifespan=lifespan
)

@app.post("/ask", response_model=AnswerResponse)
async def ask_questions(request: QuestionRequest):
    """
    Process questions against a PDF document
    
    - **documents**: URL to the PDF document
    - **questions**: List of questions to ask about the document
    """
    try:
        logger.info(f"Received request with {len(request.questions)} questions")
        
        # Process questions
        answers = await qa_system.process_questions(str(request.documents), request.questions)
        
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
        
        # Download and parse PDF
        pdf_bytes = await qa_system.download_pdf(str(request.documents))
        pages = qa_system.parse_pdf(pdf_bytes)
        
        # Create chunks and embeddings
        chunks = qa_system.smart_chunk_document(pages)
        chunks = qa_system.create_embeddings(chunks)
        
        # Process questions with detailed results
        detailed_answers = []
        for question in request.questions:
            relevant_chunks = qa_system.semantic_search(question, chunks)
            result = qa_system.enhanced_answer_generation(question, relevant_chunks)
            detailed_answers.append(result)
        
        return DetailedAnswerResponse(answers=detailed_answers)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced PDF Q&A API",
        "version": "2.0.0",
        "endpoints": {
            "/ask": "POST - Process questions (simple response)",
            "/ask-detailed": "POST - Process questions (detailed response)",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )