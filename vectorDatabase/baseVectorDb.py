from abc import ABC, abstractmethod
from typing import List, Dict
import time


class BaseVectorDatabase(ABC):
    """Enhanced Qdrant client with thread-aware storage"""
    
    def __init__(self):
        self.client = any
        self.collection_name : str
        self.embedding_dim : int 
    @abstractmethod
    def _setup_collection(self):
        """Setup collection with thread metadata"""
        pass
    @abstractmethod
    def add_chunks_by_thread(self, chunks: List[Dict], thread_id: int):
        """Add chunks with thread-specific metadata"""
        pass
    @abstractmethod
    def search_with_thread_distribution(self, query_vector: List[float], top_k: int = 8, num_threads: int = 3) -> List[List[Dict]]:
        """Search and distribute results across threads"""
        pass