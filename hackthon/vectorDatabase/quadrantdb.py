from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict
from dotenv import load_dotenv
from collections import defaultdict
import time
from hackthon.vectorDatabase.baseVectorDb import *
# Configure logging
from hackthon.llogging import *
logger = setup_logger()
# Load environment variables
load_dotenv()


class EnhancedQdrantDatabase(BaseVectorDatabase):
    """Enhanced Qdrant client with thread-aware storage"""
    
    def __init__(self, url: str, api_key: str):
        super().__init__()
        self.client = QdrantClient(url=url, api_key=api_key) if api_key else None
        self.collection_name = f"pdf_chunks_{int(time.time())}"
        self.embedding_dim = 768  # for all-mpnet-base-v2
        self._setup_collection()
    
    def _setup_collection(self):
        """Setup Qdrant collection with thread metadata"""
        if not self.client:
            return
        
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Qdrant collection '{self.collection_name}' created")
        except Exception as e:
            logger.warning(f"Collection may already exist: {e}")
    
    def add_chunks_by_thread(self, chunks: List[Dict], thread_id: int):
        """Add chunks with thread-specific metadata"""
        if not self.client:
            return
        
        try:
            points = []
            for chunk in chunks:
                # FIXED: Use pure UUID instead of prefixed string
                point_id = chunk['id']  # This is already a valid UUID
                
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=chunk['embedding'],
                        payload={
                            'text': chunk['text'],
                            'page_number': chunk['page_number'],
                            'section': chunk['section'],
                            'thread_id': thread_id,  # Thread info stored in payload
                            'chunk_id': chunk['id'],
                            'word_count': chunk['metadata']['word_count'],
                            'chunk_hash': chunk.get('chunk_hash', ''),
                            'semantic_keywords': chunk.get('keywords', [])
                        }
                    )
                )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Thread {thread_id}: Added {len(chunks)} chunks to Qdrant")
        except Exception as e:
            logger.error(f"Error adding chunks to Qdrant: {e}")
    
    def search_with_thread_distribution(self, query_vector: List[float], top_k: int = 8, num_threads: int = 3) -> List[List[Dict]]:
        """Search and distribute results across threads"""
        if not self.client:
            return [[] for _ in range(num_threads)]
        
        try:
            # Search with higher limit
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k * 2,
                with_payload=True
            )
            
            # Group by thread_id and distribute evenly
            thread_groups = defaultdict(list)
            for result in search_results:
                thread_id = result.payload.get('thread_id', 0)
                thread_groups[thread_id].append({
                    'text': result.payload['text'],
                    'page_number': result.payload['page_number'],
                    'section': result.payload['section'],
                    'similarity_score': float(result.score),
                    'chunk_hash': result.payload.get('chunk_hash', ''),
                    'metadata': {
                        'page': result.payload['page_number'],
                        'section': result.payload['section'],
                        'thread_id': thread_id,
                        'keywords': result.payload.get('semantic_keywords', [])
                    }
                })
            
            # Distribute results evenly across threads
            distributed_results = []
            chunks_per_thread = max(1, top_k // num_threads)
            
            for thread_id in range(num_threads):
                if thread_id in thread_groups:
                    thread_chunks = sorted(
                        thread_groups[thread_id],
                        key=lambda x: x['similarity_score'],
                        reverse=True
                    )[:chunks_per_thread]
                    distributed_results.append(thread_chunks)
                else:
                    # If no chunks for this thread, distribute from other threads
                    all_remaining = []
                    for tid, chunks in thread_groups.items():
                        all_remaining.extend(chunks)
                    all_remaining.sort(key=lambda x: x['similarity_score'], reverse=True)
                    distributed_results.append(all_remaining[:chunks_per_thread])
            
            return distributed_results
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return [[] for _ in range(num_threads)]
