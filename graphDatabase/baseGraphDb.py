
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseGraphDb(ABC):
    """Base Graph integration for document relationships and semantic enhancement"""
    
    def __init__(self):
        self.driver = None
    @abstractmethod
    def store_document_structure(self, chunks: List[Dict], document_id: str):
        """Store document structure and relationships in Neo4j - DISABLED"""
        pass
    @abstractmethod
    def get_enhanced_context(self, chunk_ids: List[str], question: str) -> Dict[str, Any]:
        """Get enhanced context using Graph relationships - DISABLED"""
        pass
        
    @abstractmethod
    def close(self):
        """Close Graph connection"""    
        pass