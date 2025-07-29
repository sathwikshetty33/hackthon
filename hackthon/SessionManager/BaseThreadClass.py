from abc import ABC, abstractmethod
from typing import List, Dict
from hackthon.configs import *
from hackthon.models import *
from hackthon.vectorDatabase.quadrantdb import *
from hackthon.graphDatabase.neo4j import *
from hackthon.llogging import *
logger = setup_logger()
# Load environment variables
load_dotenv()


class BaseThreadSessionSystem(ABC):
    """Optimal Multi-thread PDF Q&A System with Rate Limiting Fixes"""
    
    def __init__(self):
        self.config : Config
        
        
    @abstractmethod
    def _validate_config(self):
        """Validate configuration"""
        pass    
    @abstractmethod
    def _initialize_components(self):
        """Initialize all components (enhanced from main.py)"""
        pass
    
    @abstractmethod
    def _setup_thread_pools(self):
        """Setup optimized thread pools"""
        pass
    @abstractmethod    
    async def throttled_groq_request(self, groq_client, api_key_index: int, **kwargs):
        """Throttled Groq API request with rate limiting"""
        pass
    @abstractmethod
    async def download_pdf(self, url: str) -> bytes:
        """Download PDF with enhanced caching"""
        pass
    @abstractmethod    
    def parse_pdf(self, pdf_bytes: bytes) -> List[Dict]:
        """Parse PDF with enhanced structure extraction"""
        pass
    @abstractmethod
    def _extract_sections(self, blocks_dict) -> List[str]:
        """Extract sections"""
        pass
    @abstractmethod
    def _extract_tables(self, page) -> List[Dict]:
        """Extract tables"""
        pass
    @abstractmethod
    def _extract_lists(self, text: str) -> List[str]:
        """Extract lists"""
        pass
    @abstractmethod
    def optimal_multi_thread_chunking(self, pages: List[Dict]) -> List[Dict]:
        """Optimal multi-thread chunking using reduced threads"""
        pass
    @abstractmethod
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract semantic keywords from text"""
        pass
    @abstractmethod
    def create_embeddings_and_store(self, chunks: List[Dict], document_id: str) -> List[Dict]:
        """Create embeddings and store in Redis/Qdrant with fallback"""
        pass
        
    @abstractmethod   
    def rate_limited_answer_generation(self, question: str, chunk_groups: List[List[Dict]], document_id: str) -> List[Dict]:
        """Generate answers with proper rate limiting - FIXED"""
        pass
    @abstractmethod
    def synthesize_final_answer(self, question: str, parallel_results: List[Dict]) -> str:
        """Synthesize final answer with fallback for rate limiting"""
        pass
    @abstractmethod
    async def process_questions_rate_limited(self, pdf_url: str, questions: List[str]) -> List[str]:
        """Main processing pipeline with rate limiting and Redis caching"""
        pass
    @abstractmethod
    def close_connections(self):
        """Close all connections including Redis"""
        pass