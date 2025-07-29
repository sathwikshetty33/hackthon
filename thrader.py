# enhanced_main_optimized.py - Multi-threaded PDF Q&A with Neo4j Integration and Rate Limiting Fixes

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
from .configs import *
from .models import *
from .quadrantdb import *
from .neo4j import *
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class OptimalMultiThreadPDFQASystem:
    """Optimal Multi-thread PDF Q&A System with Rate Limiting Fixes"""
    
    def __init__(self):
        self.config = Config()
        self._validate_config()
        self._initialize_components()
        self._setup_thread_pools()
        self._document_cache = {}
        self.lock = threading.Lock()
        self.groq_clients = []
        
        # RATE LIMITING FIXES - UPDATED VALUES
        self.request_semaphore = Semaphore(1)  # REDUCED: Only 1 concurrent request
        self.last_request_time = {}  # Track last request time per API key
        self.min_request_interval = 4.0  # INCREASED: 4 seconds between requests per key
        
        # Initialize multiple Groq clients
        for i, api_key in enumerate(self.config.GROQ_API_KEYS):
            client = Groq(api_key=api_key)
            self.groq_clients.append(client)
            self.last_request_time[i] = 0
            logger.info(f"Groq client {i+1} initialized")
    
    def _validate_config(self):
        """Validate configuration"""
        if not self.config.GROQ_API_KEYS:
            raise ValueError("At least one GROQ_API_KEY is required")
        
        logger.info(f"System configured with {self.config.TOTAL_THREADS} threads using {len(self.config.GROQ_API_KEYS)} API keys")
    
    def _initialize_components(self):
        """Initialize all components (enhanced from main.py)"""
        logger.info(f"Initializing Rate-Limited Multi-Thread PDF Q&A System ({self.config.TOTAL_THREADS} threads)...")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded")
        except OSError:
            logger.warning("spaCy model not found")
            self.nlp = None
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            self.embedding_dim = 768
        except:
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            self.embedding_dim = 384
        logger.info("Embedding model loaded")
        
        # Initialize Enhanced Qdrant
        self.qdrant_db = None
        if self.config.QDRANT_URL and self.config.QDRANT_API_KEY:
            try:
                self.qdrant_db = EnhancedQdrantDatabase(
                    url=self.config.QDRANT_URL,
                    api_key=self.config.QDRANT_API_KEY
                )
                logger.info("Enhanced Qdrant database initialized")
            except Exception as e:
                logger.warning(f"Qdrant connection failed: {e}")
        
        # Initialize Neo4j Knowledge Graph (DISABLED)
        self.neo4j_kg = Neo4jKnowledgeGraph(
            uri=self.config.NEO4J_URI,
            username=self.config.NEO4J_USERNAME,
            password=self.config.NEO4J_PASSWORD
        )
        
        logger.info("All components initialized successfully!")
    
    def _setup_thread_pools(self):
        """Setup optimized thread pools"""
        self.chunk_executor = ThreadPoolExecutor(max_workers=self.config.CHUNK_THREADS)
        self.query_executor = ThreadPoolExecutor(max_workers=self.config.QUERY_THREADS)
        logger.info(f"Thread pools configured: {self.config.CHUNK_THREADS} chunking, {self.config.QUERY_THREADS} querying")
    
    async def throttled_groq_request(self, groq_client, api_key_index: int, **kwargs):
        """Throttled Groq API request with rate limiting"""
        async with self.request_semaphore:
            # Wait if needed to respect rate limits per API key
            now = time.time()
            elapsed = now - self.last_request_time.get(api_key_index, 0)
            if elapsed < self.min_request_interval:
                wait_time = self.min_request_interval - elapsed
                logger.info(f"Rate limiting: waiting {wait_time:.2f}s for API key {api_key_index + 1}")
                await asyncio.sleep(wait_time)
            
            try:
                response = groq_client.chat.completions.create(**kwargs)
                self.last_request_time[api_key_index] = time.time()
                return response
            except Exception as e:
                if "429" in str(e):
                    # Exponential backoff for rate limits
                    backoff_time = min(60, 5 * (2 ** api_key_index))
                    logger.warning(f"Rate limit hit, backing off {backoff_time}s")
                    await asyncio.sleep(backoff_time)
                    response = groq_client.chat.completions.create(**kwargs)
                    self.last_request_time[api_key_index] = time.time()
                    return response
                raise e
    
    async def download_pdf(self, url: str) -> bytes:
        """Download PDF with enhanced caching"""
        doc_hash = hashlib.md5(url.encode()).hexdigest()
        
        with self.lock:
            if doc_hash in self._document_cache:
                logger.info(f"Using cached PDF: {doc_hash[:8]}...")
                return self._document_cache[doc_hash]
        
        logger.info(f"Downloading PDF from: {url}")
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(str(url)) as response:
                    response.raise_for_status()
                    content = await response.read()
                    
                    with self.lock:
                        self._document_cache[doc_hash] = content
                    
                    logger.info(f"PDF cached: {len(content)} bytes")
                    return content
        except Exception as e:
            logger.error(f"PDF download failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    def parse_pdf(self, pdf_bytes: bytes) -> List[Dict]:
        """Parse PDF with enhanced structure extraction"""
        logger.info("Parsing PDF with enhanced structure extraction...")
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if not text.strip():
                    continue
                
                # Enhanced structure extraction
                blocks = page.get_text("dict")
                sections = self._extract_sections(blocks)
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
            logger.info(f"PDF parsed: {len(pages)} pages with enhanced structure")
            return pages
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    def _extract_sections(self, blocks_dict) -> List[str]:
        """Extract sections"""
        sections = []
        for block in blocks_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        size = span.get("size", 0)
                        flags = span.get("flags", 0)
                        
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
        """Extract tables"""
        tables = []
        try:
            table_data = page.find_tables()
            for i, table in enumerate(table_data):
                if table.extract():
                    tables.append({
                        'table_id': i,
                        'data': table.extract()[:5]
                    })
        except:
            pass
        return tables
    
    def _extract_lists(self, text: str) -> List[str]:
        """Extract lists"""
        lists = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if (re.match(r'^[•·▪▫◦‣⁃]\s+', line) or
                re.match(r'^\d+\.\s+', line) or
                re.match(r'^[a-zA-Z]\.\s+', line)):
                lists.append(line)
        return lists
    
    def optimal_multi_thread_chunking(self, pages: List[Dict]) -> List[Dict]:
        """Optimal multi-thread chunking using reduced threads"""
        logger.info(f"Creating smart chunks using {self.config.CHUNK_THREADS} threads...")
        
        # Distribute pages optimally across threads
        pages_per_thread = max(1, len(pages) // self.config.CHUNK_THREADS)
        page_groups = [
            pages[i:i + pages_per_thread]
            for i in range(0, len(pages), pages_per_thread)
        ]
        
        # Ensure we don't exceed available threads
        while len(page_groups) > self.config.CHUNK_THREADS:
            page_groups[-2].extend(page_groups[-1])
            page_groups.pop()
        
        def process_page_group_optimal(group_pages: List[Dict], thread_id: int) -> List[Dict]:
            """Optimal chunking for page group"""
            thread_chunks = []
            
            for page in group_pages:
                text = page['text']
                page_num = page['page_number']
                sections = page['sections']
                current_section = sections[0] if sections else "Document Content"
                
                # Smart paragraph-based chunking
                paragraphs = re.split(r'\n\s*\n', text)
                current_chunk = ""
                chunk_sentences = []
                
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue
                    
                    sentences = re.split(r'[.!?]+', paragraph)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence or len(sentence) < 10:
                            continue
                        
                        potential_chunk = current_chunk + " " + sentence
                        if len(potential_chunk) > self.config.CHUNK_SIZE and current_chunk:
                            # Create chunk with enhanced metadata
                            chunk_hash = hashlib.md5(current_chunk.encode()).hexdigest()[:16]
                            keywords = self._extract_keywords(current_chunk)
                            
                            thread_chunks.append({
                                'id': str(uuid.uuid4()),
                                'text': current_chunk.strip(),
                                'page_number': page_num,
                                'section': current_section,
                                'sentence_count': len(chunk_sentences),
                                'thread_id': thread_id,
                                'chunk_hash': chunk_hash,
                                'keywords': keywords,
                                'avg_similarity': 0.5,
                                'metadata': {
                                    'source': 'pdf',
                                    'page': page_num,
                                    'section': current_section,
                                    'word_count': len(current_chunk.split()),
                                    'thread_id': thread_id
                                }
                            })
                            
                            # Start new chunk with smart overlap
                            overlap_sentences = chunk_sentences[-2:] if len(chunk_sentences) >= 2 else chunk_sentences
                            current_chunk = " ".join(overlap_sentences) + " " + sentence
                            chunk_sentences = overlap_sentences + [sentence]
                        else:
                            current_chunk = potential_chunk
                            chunk_sentences.append(sentence)
                
                # Add remaining chunk
                if current_chunk.strip():
                    chunk_hash = hashlib.md5(current_chunk.encode()).hexdigest()[:16]
                    keywords = self._extract_keywords(current_chunk)
                    
                    thread_chunks.append({
                        'id': str(uuid.uuid4()),
                        'text': current_chunk.strip(),
                        'page_number': page_num,
                        'section': current_section,
                        'sentence_count': len(chunk_sentences),
                        'thread_id': thread_id,
                        'chunk_hash': chunk_hash,
                        'keywords': keywords,
                        'avg_similarity': 0.5,
                        'metadata': {
                            'source': 'pdf',
                            'page': page_num,
                            'section': current_section,
                            'word_count': len(current_chunk.split()),
                            'thread_id': thread_id
                        }
                    })
            
            logger.info(f"Thread {thread_id}: Created {len(thread_chunks)} optimal chunks")
            return thread_chunks
        
        # Process in parallel
        all_chunks = []
        future_to_thread = {
            self.chunk_executor.submit(process_page_group_optimal, group, i): i
            for i, group in enumerate(page_groups)
        }
        
        for future in as_completed(future_to_thread):
            thread_id = future_to_thread[future]
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Chunking thread {thread_id} failed: {e}")
        
        logger.info(f"Total optimal chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract semantic keywords from text"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            keywords = []
            
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                    not token.is_stop and 
                    len(token.text) > 2):
                    keywords.append(token.lemma_.lower())
            
            return list(set(keywords))[:10]  # Top 10 unique keywords
        except:
            return []
    
    def create_embeddings_and_store(self, chunks: List[Dict], document_id: str) -> List[Dict]:
        """Create embeddings and store in Qdrant"""
        logger.info("Creating embeddings and storing in databases...")
        
        # Group chunks by thread for parallel processing
        thread_chunks = defaultdict(list)
        for chunk in chunks:
            thread_id = chunk.get('thread_id', 0)
            thread_chunks[thread_id].append(chunk)
        
        def process_embeddings_for_thread(chunk_group: List[Dict], thread_id: int):
            """Create embeddings for thread group"""
            try:
                texts = [chunk['text'] for chunk in chunk_group]
                embeddings = self.embedding_model.encode(
                    texts,
                    show_progress_bar=False,
                    batch_size=32,
                    normalize_embeddings=True
                )
                
                for i, chunk in enumerate(chunk_group):
                    chunk['embedding'] = embeddings[i].tolist()
                
                # Store in Qdrant
                if self.qdrant_db:
                    self.qdrant_db.add_chunks_by_thread(chunk_group, thread_id)
                
                logger.info(f"Thread {thread_id}: Processed {len(chunk_group)} chunks")
                return chunk_group
            except Exception as e:
                logger.error(f"Embedding thread {thread_id} failed: {e}")
                return chunk_group
        
        # Process embeddings in parallel
        updated_chunks = []
        future_to_thread = {
            self.chunk_executor.submit(process_embeddings_for_thread, chunks_list, thread_id): thread_id
            for thread_id, chunks_list in thread_chunks.items()
        }
        
        for future in as_completed(future_to_thread):
            thread_id = future_to_thread[future]
            try:
                processed_chunks = future.result()
                updated_chunks.extend(processed_chunks)
            except Exception as e:
                logger.error(f"Embedding thread {thread_id} failed: {e}")
        
        # Store document structure in Neo4j (DISABLED)
        if self.neo4j_kg:
            self.neo4j_kg.store_document_structure(updated_chunks, document_id)
        
        logger.info(f"Embeddings created and stored for {len(updated_chunks)} chunks")
        return updated_chunks
    
    def rate_limited_answer_generation(self, question: str, chunk_groups: List[List[Dict]], document_id: str) -> List[Dict]:
        """Generate answers with proper rate limiting - FIXED"""
        logger.info(f"Generating answers with rate limiting using {len(chunk_groups)} threads...")
        
        def generate_answer_with_retry(context_chunks: List[Dict], group_id: int) -> Dict:
            """Generate answer with exponential backoff retry - FIXED"""
            max_retries = 3
            base_delay = 5
            
            for attempt in range(max_retries):
                try:
                    if not context_chunks:
                        return {
                            'group_id': group_id,
                            'answer': "No relevant context found.",
                            'confidence': 0.0,
                            'sources': []
                        }
                    
                    # Prepare context
                    context_parts = []
                    for i, chunk in enumerate(context_chunks[:3], 1):
                        context_parts.append(
                            f"[Context {i} - Page {chunk['page_number']}, Section: {chunk['section']}]\n{chunk['text']}"
                        )
                    
                    context_text = "\n\n".join(context_parts)
                    
                    # Enhanced prompt
                    prompt = f"""You are an expert document analyst. Provide accurate, comprehensive answers based on the provided context.

CONTEXT:
{context_text}

QUESTION: {question}

INSTRUCTIONS:
1. Analyze the question carefully
2. Use ALL provided context for your answer
3. Be precise with numbers, dates, and specific terms
4. If information is not available, state this clearly
5. Structure your response clearly and logically

ANSWER:"""
                    
                    # Use different API key for each thread
                    api_key_index = group_id % len(self.groq_clients)
                    groq_client = self.groq_clients[api_key_index]
                    
                    # Add delay between requests
                    if api_key_index in self.last_request_time:
                        elapsed = time.time() - self.last_request_time[api_key_index]
                        if elapsed < self.min_request_interval:
                            wait_time = self.min_request_interval - elapsed
                            logger.info(f"Waiting {wait_time:.2f}s for API key {api_key_index + 1}")
                            time.sleep(wait_time)
                    
                    response = groq_client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=600,  # Reduced to save tokens
                        temperature=0.1,
                        top_p=0.9
                    )
                    
                    self.last_request_time[api_key_index] = time.time()
                    answer_text = response.choices[0].message.content.strip()
                    
                    # Calculate confidence
                    avg_similarity = sum(chunk.get('similarity_score', 0) for chunk in context_chunks) / len(context_chunks)
                    confidence = min(avg_similarity * 1.2, 1.0)
                    
                    return {
                        'group_id': group_id,
                        'answer': answer_text,
                        'confidence': confidence,
                        'api_key_used': api_key_index + 1,
                        'attempt': attempt + 1,
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
                    if "429" in str(e) and attempt < max_retries - 1:
                        # FIXED: Remove question_index and use exponential backoff
                        delay = base_delay * (2 ** attempt)  # Exponential backoff: 5s, 10s, 20s
                        logger.warning(f"Rate limit hit for group {group_id}, waiting {delay}s (attempt {attempt + 1})")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Error in group {group_id} after {attempt + 1} attempts: {e}")
                        return {
                            'group_id': group_id,
                            'answer': f"Error generating answer after {attempt + 1} attempts: {str(e)}",
                            'confidence': 0.0,
                            'sources': []
                        }
            
            return {
                'group_id': group_id,
                'answer': "Failed to generate answer after all retries",
                'confidence': 0.0,
                'sources': []
            }
        
        # Generate answers with controlled concurrency
        results = []
        future_to_group = {
            self.query_executor.submit(generate_answer_with_retry, chunks, i): i
            for i, chunks in enumerate(chunk_groups)
        }
        
        for future in as_completed(future_to_group):
            group_id = future_to_group[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Answer generation failed for group {group_id}: {e}")
        
        results.sort(key=lambda x: x.get('group_id', 0))
        logger.info(f"Generated {len(results)} answers with rate limiting")
        return results
    
    def synthesize_final_answer(self, question: str, parallel_results: List[Dict]) -> str:
        """Synthesize final answer with fallback for rate limiting"""
        logger.info("Synthesizing final answer...")
        
        # Filter valid results and sort by confidence
        valid_results = [r for r in parallel_results if r['confidence'] > 0.1]
        if not valid_results:
            return "Unable to find relevant information to answer the question."
        
        valid_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # If we have high-confidence answer, use it directly
        if valid_results[0]['confidence'] > 0.8:
            return valid_results[0]['answer']
        
        # Try synthesis with rate limiting protection
        try:
            # Check if we have recent API usage
            now = time.time()
            last_synthesis_key = 0  # Use first API key for synthesis
            if last_synthesis_key in self.last_request_time:
                elapsed = now - self.last_request_time[last_synthesis_key]
                if elapsed < self.min_request_interval:
                    logger.warning("Skipping synthesis due to rate limiting, using best single answer")
                    return valid_results[0]['answer']
            
            answers_text = "\n\n".join([
                f"Response {i+1} (confidence: {r['confidence']:.2f}): {r['answer']}"
                for i, r in enumerate(valid_results[:2])  # Only use top 2 to save tokens
            ])
            
            synthesis_prompt = f"""Synthesize these responses into one coherent answer:

QUESTION: {question}

RESPONSES:
{answers_text}

Provide a single, accurate answer combining the best information from both responses.

ANSWER:"""
            
            # Add delay before synthesis
            if last_synthesis_key in self.last_request_time:
                elapsed = time.time() - self.last_request_time[last_synthesis_key]
                if elapsed < self.min_request_interval:
                    time.sleep(self.min_request_interval - elapsed)
            
            response = self.groq_clients[0].chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=400,  # Reduced token usage
                temperature=0.1
            )
            
            self.last_request_time[last_synthesis_key] = time.time()
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback to best single answer
            return valid_results[0]['answer']
    
    async def process_questions_rate_limited(self, pdf_url: str, questions: List[str]) -> List[str]:
        """Main processing pipeline with rate limiting"""
        logger.info(f"Processing {len(questions)} questions with rate limiting...")
        start_time = time.time()
        
        try:
            # Generate document ID
            document_id = hashlib.md5(pdf_url.encode()).hexdigest()
            
            # Step 1: Download and parse PDF
            pdf_bytes = await self.download_pdf(pdf_url)
            pages = self.parse_pdf(pdf_bytes)
            
            # Step 2: Multi-thread chunking
            chunks = self.optimal_multi_thread_chunking(pages)
            
            # Step 3: Create embeddings and store
            chunks = self.create_embeddings_and_store(chunks, document_id)
            
            # Step 4: Process each question with rate limiting
            final_answers = []
            
            for question_index, question in enumerate(questions):  # ADDED question_index
                question_start = time.time()
                
                # Get question embedding
                question_embedding = self.embedding_model.encode([question], normalize_embeddings=True)[0].tolist()
                
                # Search with thread distribution
                chunk_groups = []
                if self.qdrant_db:
                    chunk_groups = self.qdrant_db.search_with_thread_distribution(
                        query_vector=question_embedding,
                        top_k=self.config.MAX_CONTEXT_CHUNKS,
                        num_threads=self.config.QUERY_THREADS
                    )
                
                # Generate answers with rate limiting
                parallel_results = self.rate_limited_answer_generation(question, chunk_groups, document_id)
                
                # Synthesize final answer
                final_answer = self.synthesize_final_answer(question, parallel_results)
                final_answers.append(final_answer)
                
                question_time = time.time() - question_start
                logger.info(f"Question {question_index + 1} processed in {question_time:.2f}s")
            
            total_time = time.time() - start_time
            logger.info(f"Rate-limited processing completed in {total_time:.2f}s")
            
            return final_answers
            
        except Exception as e:
            logger.error(f"Rate-limited processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def close_connections(self):
        """Close all connections"""
        if self.neo4j_kg:
            self.neo4j_kg.close()
        
        if hasattr(self, 'chunk_executor'):
            self.chunk_executor.shutdown(wait=True)
        if hasattr(self, 'query_executor'):
            self.query_executor.shutdown(wait=True)
        
        logger.info("All connections closed")
