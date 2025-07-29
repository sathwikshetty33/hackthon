# redis_session.py - Fixed Redis Session Manager for PDF Q&A System

import redis
import json
import pickle
import logging
from typing import List, Dict, Any, Optional
import hashlib
import time
from datetime import datetime, timedelta
import gzip
from urllib.parse import urlparse, quote_plus
import threading

logger = logging.getLogger(__name__)

class RedisSessionManager:
    """Redis Session Manager for caching chunks and embeddings with Upstash support"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, password: str = None, 
             db: int = 0, decode_responses: bool = False, max_retries: int = 3, 
             redis_url: str = None, ssl: bool = False, connection_timeout: int = 10):
        """
        Initialize Redis connection with Upstash and standard Redis support
        """
        self.max_retries = max_retries
        self.connection_pool = None
        self.redis_client = None
        self.is_available = False
        self.connection_timeout = connection_timeout

        try:
            # Prioritize direct connection parameters over URL
            if redis_url and not (host != 'localhost' or port != 6379 or password or ssl):
                # Only use URL if no direct parameters were provided
                logger.info(f"Connecting to Redis using URL")
                
                # Parse and fix URL for Upstash compatibility
                parsed_url = urlparse(redis_url)
                
                # Handle Upstash URL formatting issues
                if parsed_url.hostname and 'upstash.io' in parsed_url.hostname:
                    # Extract components manually to avoid IDNA issues
                    hostname = parsed_url.hostname
                    url_password = parsed_url.password
                    
                    if not url_password:
                        raise ValueError("No password found in Redis URL")
                    
                    logger.info(f"Connecting to Upstash Redis at {hostname}")
                    
                    # Use manual connection parameters with reduced timeouts
                    self.redis_client = redis.Redis(
                        host=hostname,
                        port=6380,  # Standard Upstash SSL port
                        password=url_password,
                        ssl=True,
                        ssl_cert_reqs=None,
                        ssl_check_hostname=False,
                        decode_responses=decode_responses,
                        socket_connect_timeout=connection_timeout,
                        socket_timeout=connection_timeout,
                        retry_on_timeout=True,
                        health_check_interval=60,
                        retry_on_error=[redis.ConnectionError, redis.TimeoutError]
                    )
                else:
                    # Standard Redis URL with timeout
                    self.redis_client = redis.from_url(
                        redis_url,
                        decode_responses=decode_responses,
                        socket_connect_timeout=connection_timeout,
                        socket_timeout=connection_timeout,
                        retry_on_timeout=True,
                        health_check_interval=30,
                        ssl_cert_reqs=None if parsed_url.scheme == 'rediss' else None,
                        retry_on_error=[redis.ConnectionError, redis.TimeoutError]
                    )
            else:
                # Direct connection parameters (preferred for Upstash)
                logger.info(f"Connecting to Redis at {host}:{port}")
                connection_kwargs = {
                    'host': host,
                    'port': port,
                    'db': db,
                    'decode_responses': decode_responses,
                    'socket_connect_timeout': connection_timeout,
                    'socket_timeout': connection_timeout,
                    'retry_on_timeout': True,
                    'health_check_interval': 60
                }
                
                if password:
                    connection_kwargs['password'] = password
                
                if ssl:
                    connection_kwargs['ssl'] = True
                    connection_kwargs['ssl_cert_reqs'] = None
                    connection_kwargs['ssl_check_hostname'] = False
                
                # Add specific settings for Upstash
                if host and 'upstash.io' in host:
                    logger.info("Detected Upstash Redis, applying optimized settings")
                    connection_kwargs['retry_on_error'] = [redis.ConnectionError, redis.TimeoutError]
                
                self.redis_client = redis.Redis(**connection_kwargs)

            # Test connection with timeout and exponential backoff
            self._test_connection_with_timeout()

        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            self.is_available = False

    def _test_connection_with_timeout(self):
        """Test connection with proper timeout handling"""
        def connection_test():
            try:
                for attempt in range(self.max_retries):
                    try:
                        logger.info(f"Testing Redis connection (attempt {attempt + 1}/{self.max_retries})")
                        
                        # Use a shorter timeout for the ping test
                        result = self.redis_client.ping()
                        if result:
                            self.is_available = True
                            logger.info("Redis connection established successfully")
                            
                            # Run quick verification test
                            test_results = self._quick_connection_test()
                            if test_results['read_write_test']:
                                logger.info("Redis read/write test passed")
                            else:
                                logger.warning("Redis read/write test failed, but connection is established")
                            return
                        
                    except (redis.ConnectionError, redis.TimeoutError) as ce:
                        if attempt < self.max_retries - 1:
                            wait_time = min(2 ** attempt, 5)  # Cap at 5 seconds
                            logger.warning(f"Redis connection attempt {attempt + 1} failed: {ce}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Redis connection failed after {self.max_retries} attempts: {ce}")
                            self.is_available = False
                            
                    except Exception as e:
                        logger.error(f"Unexpected error during connection test: {e}")
                        self.is_available = False
                        return

            except Exception as e:
                logger.error(f"Connection test thread failed: {e}")
                self.is_available = False

        # Run connection test in a separate thread with timeout
        connection_thread = threading.Thread(target=connection_test, daemon=True)
        connection_thread.start()
        connection_thread.join(timeout=self.connection_timeout * 2)  # Give it double the connection timeout
        
        if connection_thread.is_alive():
            logger.error(f"Redis connection test timed out after {self.connection_timeout * 2} seconds")
            self.is_available = False

    def _quick_connection_test(self) -> Dict[str, Any]:
        """Quick connection test with timeout"""
        test_results = {
            'ping_successful': False,
            'read_write_test': False,
            'error_details': None
        }
        
        try:
            # Test ping with timeout
            ping_result = self.redis_client.ping()
            test_results['ping_successful'] = ping_result
            
            # Test read/write operations with timeout
            test_key = f"test_key_{int(time.time())}"
            test_value = "test_value_123"
            
            # Write test
            self.redis_client.set(test_key, test_value, ex=60)
            
            # Read test
            retrieved_value = self.redis_client.get(test_key)
            if retrieved_value:
                if isinstance(retrieved_value, bytes):
                    retrieved_value = retrieved_value.decode('utf-8')
                
                if retrieved_value == test_value:
                    test_results['read_write_test'] = True
            
            # Clean up
            self.redis_client.delete(test_key)
            
        except Exception as e:
            test_results['error_details'] = str(e)
            logger.error(f"Connection test failed: {e}")
        
        return test_results
    
    def _generate_cache_key(self, prefix: str, identifier: str) -> str:
        """Generate standardized cache key"""
        return f"{prefix}:{hashlib.md5(identifier.encode()).hexdigest()}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for Redis storage with compression for large data"""
        try:
            # Use pickle for better performance with large embeddings
            pickled_data = pickle.dumps(data)
            
            # Compress large data (>1KB) to save memory in cloud Redis
            if len(pickled_data) > 1024:
                return gzip.compress(pickled_data)
            else:
                return pickled_data
                
        except Exception as e:
            logger.error(f"Data serialization failed: {e}")
            # Fallback to JSON
            return json.dumps(data, default=str).encode('utf-8')
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from Redis with decompression support"""
        try:
            # Try to decompress first
            try:
                decompressed_data = gzip.decompress(data)
                return pickle.loads(decompressed_data)
            except:
                # If decompression fails, try direct pickle
                return pickle.loads(data)
                
        except Exception:
            try:
                # Fallback to JSON
                return json.loads(data.decode('utf-8'))
            except Exception as e:
                logger.error(f"Data deserialization failed: {e}")
                return None
    
    def is_connected(self) -> bool:
        """Check if Redis is currently connected"""
        if not self.is_available or not self.redis_client:
            return False
        
        try:
            return self.redis_client.ping()
        except:
            return False
    
    def store_chunks(self, chunks: List[Dict], document_id: str, expiry_hours: int = 24) -> bool:
        """Store document chunks in Redis with expiry"""
        if not self.is_available:
            logger.warning("Redis not available, skipping chunk storage")
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
            # Don't mark as unavailable for single operation failures
        
        return False
    
    def get_chunks(self, document_id: str) -> Optional[List[Dict]]:
        """Retrieve document chunks from Redis"""
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
        
        return None
    
    def store_embeddings(self, embeddings: List[List[float]], document_id: str, expiry_hours: int = 24) -> bool:
        """Store embeddings separately for memory efficiency"""
        if not self.is_available:
            logger.warning("Redis not available, skipping embedding storage")
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
        
        return False
    
    def get_embeddings(self, document_id: str) -> Optional[List[List[float]]]:
        """Retrieve embeddings from Redis"""
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
        
        return None
    
    def store_question_cache(self, question: str, answer: str, document_id: str, expiry_hours: int = 6) -> bool:
        """Cache question-answer pairs for faster retrieval"""
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
        
        return False
    
    def get_question_cache(self, question: str, document_id: str) -> Optional[str]:
        """Retrieve cached answer for question"""
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
        
        return None
    
    def get_cache_stats(self, document_id: str) -> Dict[str, Any]:
        """Get cache statistics for a document"""
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
        
        return stats
    
    def clear_document_cache(self, document_id: str) -> bool:
        """Clear all cached data for a document"""
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
            return False
    
    def test_upstash_connection(self) -> Dict[str, Any]:
        """Comprehensive connection test for Upstash Redis"""
        test_results = {
            'connection_successful': False,
            'ping_successful': False,
            'read_write_test': False,
            'error_details': None,
            'server_info': None
        }
        
        if not self.redis_client:
            test_results['error_details'] = "Redis client not initialized"
            return test_results
        
        try:
            # Test basic connection
            ping_result = self.redis_client.ping()
            test_results['ping_successful'] = ping_result
            test_results['connection_successful'] = True
            
            # Test read/write operations
            test_key = f"upstash_test_key_{int(time.time())}"
            test_value = "test_value_123"
            
            # Write test
            self.redis_client.set(test_key, test_value, ex=60)  # 60 second expiry
            
            # Read test
            retrieved_value = self.redis_client.get(test_key)
            if retrieved_value:
                if isinstance(retrieved_value, bytes):
                    retrieved_value = retrieved_value.decode('utf-8')
                
                if retrieved_value == test_value:
                    test_results['read_write_test'] = True
            
            # Clean up test key
            self.redis_client.delete(test_key)
            
            # Get server info if available
            try:
                server_info = self.redis_client.info()
                test_results['server_info'] = {
                    'redis_version': server_info.get('redis_version', 'unknown'),
                    'used_memory_human': server_info.get('used_memory_human', 'unknown'),
                    'connected_clients': server_info.get('connected_clients', 'unknown')
                }
            except Exception as info_e:
                logger.warning(f"Could not retrieve server info: {info_e}")
            
        except Exception as e:
            test_results['error_details'] = str(e)
            logger.error(f"Upstash connection test failed: {e}")
        
        return test_results
    
    def close(self):
        """Close Redis connections"""
        try:
            if self.redis_client:
                self.redis_client.close()
            if self.connection_pool:
                self.connection_pool.disconnect()
            logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")


# Alternative: Fast-fail Redis connection for debugging
class FastFailRedisSessionManager(RedisSessionManager):
    """Redis Session Manager that fails fast if connection issues occur"""
    
    def __init__(self, *args, **kwargs):
        # Set very short timeout for debugging
        kwargs['connection_timeout'] = kwargs.get('connection_timeout', 5)
        super().__init__(*args, **kwargs)
    
    def _test_connection_with_timeout(self):
        """Fast-fail connection test"""
        try:
            logger.info("Testing Redis connection with fast-fail mode...")
            
            # Single attempt with short timeout
            result = self.redis_client.ping()
            if result:
                self.is_available = True
                logger.info("Redis connection established successfully")
            else:
                logger.error("Redis ping failed")
                self.is_available = False
                
        except Exception as e:
            logger.error(f"Redis connection failed immediately: {e}")
            self.is_available = False


# Helper functions
def create_upstash_session_manager(endpoint: str, password: str, decode_responses: bool = False, 
                                 fast_fail: bool = False) -> RedisSessionManager:
    """
    Create RedisSessionManager for Upstash Redis with direct parameters
    
    Args:
        endpoint: Upstash endpoint (e.g., 'bursting-elk-44588.upstash.io')
        password: Redis password
        decode_responses: Whether to decode responses
        fast_fail: Use fast-fail mode for debugging
    
    Returns:
        RedisSessionManager instance
    """
    manager_class = FastFailRedisSessionManager if fast_fail else RedisSessionManager
    
    return manager_class(
        host=endpoint,
        port=6380,
        password=password,
        ssl=True,
        decode_responses=decode_responses,
        connection_timeout=5 if fast_fail else 10
    )

def format_upstash_url(endpoint: str, password: str, port: int = 6380) -> str:
    """
    Format Upstash Redis URL properly
    
    Args:
        endpoint: Upstash endpoint (e.g., 'bursting-elk-44588.upstash.io')
        password: Redis password
        port: Redis port (default 6380 for SSL)
    
    Returns:
        Properly formatted Redis URL
    """
    encoded_password = quote_plus(password)
    return f"rediss://:{encoded_password}@{endpoint}:{port}"

def test_redis_connection(redis_url: str = None, endpoint: str = None, password: str = None):
    """Test Redis connection with proper error handling and debugging"""
    
    if redis_url:
        print(f"Testing Redis connection to URL: {redis_url}")
        session_manager = RedisSessionManager(redis_url=redis_url, connection_timeout=10)
    elif endpoint and password:
        print(f"Testing Redis connection to endpoint: {endpoint}")
        session_manager = create_upstash_session_manager(endpoint, password, fast_fail=True)
    else:
        print("Error: Please provide either redis_url or both endpoint and password")
        return
    
    # Wait a moment for connection to establish
    time.sleep(2)
    
    if session_manager.is_available:
        print("✅ Redis connection successful!")
        
        # Run comprehensive test
        test_results = session_manager.test_upstash_connection()
        print(f"Connection test results: {test_results}")
        
        # Test basic operations
        test_chunks = [
            {"id": 1, "content": "Test chunk 1", "metadata": {"page": 1}},
            {"id": 2, "content": "Test chunk 2", "metadata": {"page": 1}}
        ]
        
        document_id = "test_document_123"
        
        # Test chunk storage
        if session_manager.store_chunks(test_chunks, document_id):
            print("✅ Chunk storage test passed")
            
            # Test chunk retrieval
            retrieved_chunks = session_manager.get_chunks(document_id)
            if retrieved_chunks and len(retrieved_chunks) == len(test_chunks):
                print("✅ Chunk retrieval test passed")
            else:
                print("❌ Chunk retrieval test failed")
        else:
            print("❌ Chunk storage test failed")
        
        # Clean up
        session_manager.clear_document_cache(document_id)
        
    else:
        print("❌ Redis connection failed")
        print("Debugging suggestions:")
        print("1. Check if your Upstash Redis instance is active")
        print("2. Verify the endpoint and password are correct")
        print("3. Check network connectivity")
        print("4. Try using the fast-fail mode for quicker debugging")
    
    session_manager.close()

