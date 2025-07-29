# redis_session.py - Redis Session Manager for PDF Q&A System

import redis
import json
import pickle
import logging
from typing import List, Dict, Any, Optional
import hashlib
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RedisSessionManager:
    """Redis Session Manager for caching chunks and embeddings"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, password: str = None, 
                 db: int = 0, decode_responses: bool = False, max_retries: int = 3):
        """
        Initialize Redis connection with retry logic
        
        Args:
            host: Redis host
            port: Redis port
            password: Redis password
            db: Redis database number
            decode_responses: Whether to decode responses
            max_retries: Maximum connection retries
        """
        self.max_retries = max_retries
        self.connection_pool = None
        self.redis_client = None
        self.is_available = False
        
        try:
            # Create connection pool for better performance
            self.connection_pool = redis.ConnectionPool(
                host=host,
                port=port,
                password=password,
                db=db,
                decode_responses=decode_responses,
                max_connections=20,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Initialize Redis client
            self.redis_client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            self.redis_client.ping()
            self.is_available = True
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.is_available = False
    
    def _generate_cache_key(self, prefix: str, identifier: str) -> str:
        """Generate standardized cache key"""
        return f"{prefix}:{hashlib.md5(identifier.encode()).hexdigest()}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for Redis storage"""
        try:
            return pickle.dumps(data)
        except Exception as e:
            logger.error(f"Data serialization failed: {e}")
            return json.dumps(data, default=str).encode('utf-8')
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from Redis"""
        try:
            return pickle.loads(data)
        except Exception:
            try:
                return json.loads(data.decode('utf-8'))
            except Exception as e:
                logger.error(f"Data deserialization failed: {e}")
                return None
    
    def store_chunks(self, chunks: List[Dict], document_id: str, expiry_hours: int = 24) -> bool:
        """
        Store document chunks in Redis with expiry
        
        Args:
            chunks: List of chunk dictionaries
            document_id: Unique document identifier
            expiry_hours: Cache expiry in hours
            
        Returns:
            bool: Success status
        """
        if not self.is_available:
            return False
        
        try:
            cache_key = self._generate_cache_key("chunks", document_id)
            serialized_chunks = self._serialize_data(chunks)
            
            # Store with expiry
            expiry_seconds = expiry_hours * 3600
            result = self.redis_client.setex(
                name=cache_key,
                time=expiry_seconds,
                value=serialized_chunks
            )
            
            if result:
                # Store metadata
                metadata = {
                    'document_id': document_id,
                    'chunk_count': len(chunks),
                    'stored_at': datetime.now().isoformat(),
                    'expires_at': (datetime.now() + timedelta(hours=expiry_hours)).isoformat()
                }
                metadata_key = self._generate_cache_key("metadata", document_id)
                self.redis_client.setex(
                    name=metadata_key,
                    time=expiry_seconds,
                    value=self._serialize_data(metadata)
                )
                
                logger.info(f"Stored {len(chunks)} chunks in Redis for document {document_id[:8]}...")
                return True
            
        except Exception as e:
            logger.error(f"Failed to store chunks in Redis: {e}")
            self.is_available = False
        
        return False
    
    def get_chunks(self, document_id: str) -> Optional[List[Dict]]:
        """
        Retrieve document chunks from Redis
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            List of chunks or None if not found
        """
        if not self.is_available:
            return None
        
        try:
            cache_key = self._generate_cache_key("chunks", document_id)
            serialized_chunks = self.redis_client.get(cache_key)
            
            if serialized_chunks:
                chunks = self._deserialize_data(serialized_chunks)
                if chunks:
                    logger.info(f"Retrieved {len(chunks)} chunks from Redis for document {document_id[:8]}...")
                    return chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks from Redis: {e}")
            self.is_available = False
        
        return None
    
    def store_embeddings(self, embeddings: List[List[float]], document_id: str, expiry_hours: int = 24) -> bool:
        """
        Store embeddings separately for memory efficiency
        
        Args:
            embeddings: List of embedding vectors
            document_id: Unique document identifier
            expiry_hours: Cache expiry in hours
            
        Returns:
            bool: Success status
        """
        if not self.is_available:
            return False
        
        try:
            cache_key = self._generate_cache_key("embeddings", document_id)
            serialized_embeddings = self._serialize_data(embeddings)
            
            expiry_seconds = expiry_hours * 3600
            result = self.redis_client.setex(
                name=cache_key,
                time=expiry_seconds,
                value=serialized_embeddings
            )
            
            if result:
                logger.info(f"Stored {len(embeddings)} embeddings in Redis for document {document_id[:8]}...")
                return True
            
        except Exception as e:
            logger.error(f"Failed to store embeddings in Redis: {e}")
            self.is_available = False
        
        return False
    
    def get_embeddings(self, document_id: str) -> Optional[List[List[float]]]:
        """
        Retrieve embeddings from Redis
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            List of embeddings or None if not found
        """
        if not self.is_available:
            return None
        
        try:
            cache_key = self._generate_cache_key("embeddings", document_id)
            serialized_embeddings = self.redis_client.get(cache_key)
            
            if serialized_embeddings:
                embeddings = self._deserialize_data(serialized_embeddings)
                if embeddings:
                    logger.info(f"Retrieved {len(embeddings)} embeddings from Redis for document {document_id[:8]}...")
                    return embeddings
            
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings from Redis: {e}")
            self.is_available = False
        
        return None
    
    def store_question_cache(self, question: str, answer: str, document_id: str, expiry_hours: int = 6) -> bool:
        """
        Cache question-answer pairs for faster retrieval
        
        Args:
            question: The question text
            answer: The generated answer
            document_id: Document identifier
            expiry_hours: Cache expiry in hours
            
        Returns:
            bool: Success status
        """
        if not self.is_available:
            return False
        
        try:
            question_hash = hashlib.md5(f"{document_id}:{question}".encode()).hexdigest()
            cache_key = self._generate_cache_key("qa", question_hash)
            
            qa_data = {
                'question': question,
                'answer': answer,
                'document_id': document_id,
                'cached_at': datetime.now().isoformat()
            }
            
            expiry_seconds = expiry_hours * 3600
            result = self.redis_client.setex(
                name=cache_key,
                time=expiry_seconds,
                value=self._serialize_data(qa_data)
            )
            
            if result:
                logger.info(f"Cached Q&A for document {document_id[:8]}...")
                return True
            
        except Exception as e:
            logger.error(f"Failed to cache Q&A: {e}")
            self.is_available = False
        
        return False
    
    def get_question_cache(self, question: str, document_id: str) -> Optional[str]:
        """
        Retrieve cached answer for question
        
        Args:
            question: The question text
            document_id: Document identifier
            
        Returns:
            Cached answer or None if not found
        """
        if not self.is_available:
            return None
        
        try:
            question_hash = hashlib.md5(f"{document_id}:{question}".encode()).hexdigest()
            cache_key = self._generate_cache_key("qa", question_hash)
            
            serialized_data = self.redis_client.get(cache_key)
            if serialized_data:
                qa_data = self._deserialize_data(serialized_data)
                if qa_data and isinstance(qa_data, dict):
                    logger.info(f"Retrieved cached answer for document {document_id[:8]}...")
                    return qa_data.get('answer')
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached Q&A: {e}")
            self.is_available = False
        
        return None
    
    def get_cache_stats(self, document_id: str) -> Dict[str, Any]:
        """
        Get cache statistics for a document
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'redis_available': self.is_available,
            'chunks_cached': False,
            'embeddings_cached': False,
            'metadata': None
        }
        
        if not self.is_available:
            return stats
        
        try:
            # Check chunks
            chunks_key = self._generate_cache_key("chunks", document_id)
            stats['chunks_cached'] = self.redis_client.exists(chunks_key) > 0
            
            # Check embeddings
            embeddings_key = self._generate_cache_key("embeddings", document_id)
            stats['embeddings_cached'] = self.redis_client.exists(embeddings_key) > 0
            
            # Get metadata
            metadata_key = self._generate_cache_key("metadata", document_id)
            metadata_data = self.redis_client.get(metadata_key)
            if metadata_data:
                stats['metadata'] = self._deserialize_data(metadata_data)
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            self.is_available = False
        
        return stats
    
    def clear_document_cache(self, document_id: str) -> bool:
        """
        Clear all cached data for a document
        
        Args:
            document_id: Document identifier
            
        Returns:
            bool: Success status
        """
        if not self.is_available:
            return False
        
        try:
            keys_to_delete = [
                self._generate_cache_key("chunks", document_id),
                self._generate_cache_key("embeddings", document_id),
                self._generate_cache_key("metadata", document_id)
            ]
            
            deleted_count = self.redis_client.delete(*keys_to_delete)
            logger.info(f"Cleared {deleted_count} cache entries for document {document_id[:8]}...")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to clear document cache: {e}")
            self.is_available = False
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform Redis health check
        
        Returns:
            Dictionary with health status
        """
        health_status = {
            'redis_available': self.is_available,
            'connection_pool_created': self.connection_pool is not None,
            'ping_successful': False,
            'error': None
        }
        
        if self.redis_client:
            try:
                response = self.redis_client.ping()
                health_status['ping_successful'] = response
                self.is_available = True
            except Exception as e:
                health_status['error'] = str(e)
                self.is_available = False
        
        return health_status
    
    def close(self):
        """Close Redis connections"""
        try:
            if self.connection_pool:
                self.connection_pool.disconnect()
            logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")