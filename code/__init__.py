"""
Hybrid RAG System Package
"""

__version__ = "1.0.0"
__author__ = "BITS Pilani Student"

# Minimal exports to avoid circular imports
# Import only core modules, not evaluation which lives in separate folder
from .data_collection import DataPreprocessor, WikipediaDataCollector, create_fixed_urls_file
from .rag_system import HybridRAGSystem, DenseRetriever, SparseRetriever, ReciprocalRankFusion

__all__ = [
    'DataPreprocessor',
    'WikipediaDataCollector',
    'create_fixed_urls_file',
    'HybridRAGSystem',
    'DenseRetriever',
    'SparseRetriever',
    'ReciprocalRankFusion',
]
